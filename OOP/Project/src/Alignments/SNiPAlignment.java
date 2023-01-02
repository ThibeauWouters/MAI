package alignments;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * This is a class implementing the SNiP view of an alignment. It is a subclass of StandardAlignment
 * and overrides some of the methods of StandardAlignment.
 * 
 * @author thibe
 */
public class SNiPAlignment extends StandardAlignment {

	/* Variables */ 
	/**
	 * referenceGenome is a String holding the original representation of the reference genome used
	 * to construct the SNiP view of an alignment. Useful in case we want to revert sequences of differences
	 * between genomes back to the original genome sequences. Note that the current implementation does not
	 * allow the reference genome to be changed, which is why this field is final.
	 */
	private final String referenceGenome;
	/**
	 * referenceID contains the identifier of the reference genome.
	 */
	private final String referenceID;

	/*  Constructors */

	/**
	 * Constructor creating a SNiP alignment from reading a .fasta file.
	 * 
	 * @param fasta A FastaContents object, which holds the contents of a .fasta file.
	 * @param referenceID The ID of the reference genome.
	 */
	public SNiPAlignment(FastaContents fasta, String referenceID) {
		// Create standard alignment first
		super(fasta);

		// If the reference ID is not in the alignment, change to default (first) ID
		if (!(this.containsIdentifier(referenceID))) {
			referenceID = this.getIdentifiers().get(0);
		}
		this.referenceID     = referenceID;
		this.referenceGenome = this.getGenome(referenceID);

		// Edit from standard alignment to SNiP, get "difference" of all sequences
		this.replaceDifferences();
	}

	/**
	 * Constructor creating a SNiP alignment from a FastaContents object, and use the first reference ID as the
	 * default value.
	 * 
	 * @param fasta A FastaContents object, which holds the contents of a .fasta file.
	 */
	public SNiPAlignment(FastaContents fasta) {		
		this(fasta, fasta.getIdentifiers().get(0));
	}

	/**
	 * Constructor taking a HashMap, representing an alignment, and the ID of the reference genome.
	 * 
	 * @param hmap A HashMap containing (ID, genome)-pairs to be stored as alingment.
	 * @param referenceID The ID of the reference genome to create the SNiP view.
	 */
	public SNiPAlignment(HashMap<String, String> hmap, String referenceID) {		
		// Create standard alignment first
		super(hmap);

		// If the reference ID is not in the alignment, change to default (first) ID
		if (!(this.containsIdentifier(referenceID))) {
			referenceID = this.getIdentifiers().get(0);
		}
		this.referenceID     = referenceID;
		this.referenceGenome = super.getGenome(referenceID);

		// Edit from standard alignment to SNiP, get "difference" of all sequences
		this.replaceDifferences();
	}

	/**
	 * Constructor taking a HashMap, representing an alignment. The reference genome taken by default
	 * is the first one present in the HashMap.
	 * 
	 * @param hmap A HashMap containing (ID, genome)-pairs to be stored as alingment.
	 */
	public SNiPAlignment(HashMap<String, String> hmap) {
		this(hmap, hmap.keySet().iterator().next());
	}

	/**
	 * Constructor taking a StandardAlignment object and the ID of the reference genome. 
	 * @param sa
	 * @param referenceID
	 */
	public SNiPAlignment(StandardAlignment sa, String referenceID) {
		// Create standard alignment first
		super(sa);

		// If the reference ID is not in the alignment, change to default (first) ID
		if (!(this.containsIdentifier(referenceID))) {
			referenceID = this.getIdentifiers().get(0);
		}
		this.referenceID     = referenceID;
		this.referenceGenome = sa.getGenome(referenceID);

		// Edit from standard alignment to SNiP, get "difference" of all sequences
		this.replaceDifferences();
	}

	/**
	 * Constructor taking a StandardAlignment object, and uses the first ID of this alignment
	 * as the reference genome to construct the SNiP view.
	 *  
	 * @param sa A StandardAlignment object from which we want to construct the SNiP view.
	 */
	public SNiPAlignment(StandardAlignment sa) {
		this(sa, sa.getIdentifiers().get(0));
	}

	/**
	 * Constructor taking another SNiPAlignment object, and creating a new copy of that object.
	 * @param snip A SNiPAlignment object which we wish to copy. 
	 */
	public SNiPAlignment(SNiPAlignment snip) {
		// Store the alignment
		this.setAlignment(snip.copyAlignment());
		// Save reference genome and ID
		this.referenceGenome = snip.getReferenceGenome();
		this.referenceID     = snip.getReferenceID();
	}


	/* Methods */

	/**
	 * Method which creates the "difference view" of a given genome when compared to another reference genome
	 * 
	 * @param genome The genome of which we want to create the "difference view"
	 * @param referenceGenome The reference genome of the SNiP alignment, with which we compare the genome param.
	 * @return The difference view of a genome, i.e., a representation where dots denote identical nucleotides 
	 * appearing in the reference genome.
	 */
	public static String getGenomeDifference(String genome, String referenceGenome) {
		// Construct a string containing the differences, initialize as an empty string
		String differences = "";
		try {
			// Iterate over all characters. If identical, add dot, otherwise add character of "genome" param
			for(int i = 0; i<genome.length(); i++) {
				if(referenceGenome.charAt(i) == genome.charAt(i)) {
					differences += ".";
				} else {
					differences += genome.charAt(i);
				}
			}
		} catch (IndexOutOfBoundsException e) {
			System.out.println("Index out of bounds. Genome and reference genome should have same length.");
			e.printStackTrace();
		}		
		return differences;
	}

	/**
	 * Create a "difference view" of the provided genome, by comparing it with the reference genome stored as an
	 * instance variable to this object.
	 * 
	 * @param genome The genome of which we want to create the "difference view"
	 * @return The difference view of a genome, i.e., a representation where dots denote identical nucleotides 
	 * appearing in the reference genome.
	 */
	public String getGenomeDifference(String genome) {
		return getGenomeDifference(genome, this.referenceGenome);
	}

	/**
	 * Iterates over all the genomes stored in the alignment, and replaces them with their difference view.
	 * Used frequently in the constructors of SNiPAlignment to change from standard to SNiP view.
	 */
	public void replaceDifferences() {
		// Iterate over all (id, seq)-pairs in the alignment, gets the difference, and save with that ID
		for (String id : this.getIdentifiers()) {
			String genome     = this.getGenome(id);
			String difference = this.getGenomeDifference(genome);
			this.replaceGenome(id, difference);
		}
	}

	/**
	 * Given the difference view of a genome and a reference genome, replace all the dots in the difference view
	 * with the nucleotides appearing in reference genome, to go back to the "standard" view.
	 * 
	 * @param difference A difference view of a genome sequence
	 * @param referenceGenome The reference genome used to construct the difference views.
	 * @return The standard view of the difference sequence as compared to the reference genome.
	 */
	public static String getOriginalGenome(String difference, String referenceGenome) {
		// Construct a string containing the original sequence, initialize it with an empty string
		String original = "";
		// Iterate over all characters. If a dot, add char of reference genome, otherwise add char of "difference" string
		try {
			for (int i = 0; i<referenceGenome.length(); i++) {
				if(difference.charAt(i) == '.') {
					original += referenceGenome.charAt(i);
				} else {
					original += difference.charAt(i);
				}
			}
		} catch (IndexOutOfBoundsException e) {
			System.out.println("Index out of bounds. Difference and reference genome should have same length.");
			e.printStackTrace();
		}		
		return original;
	}

	/**
	 * Given the difference view of a genome, replace all the dots in the difference view with the 
	 * nucleotides appearing in reference genome stored in this Object, and go back to "standard" view.
	 * 
	 * @param difference A difference view of a genome sequence
	 * @param referenceGenome The reference genome used to construct the difference views.
	 * @return The standard view of the difference sequence as compared to the reference genome.
	 */
	public String getOriginalGenome(String difference) {
		return getOriginalGenome(this.referenceGenome, difference);
	}

	@Override
	/**
	 * See StandardAlignment. This method is overriden from the superclass since we assume the sequences
	 * to be represented in the "standard" view, such that we first compute the difference view of the
	 * replaced versions of the genome sequences before storing them. 
	 */
	public void replaceSequenceGenome(String identifier, String oldSequence, String newSequence) {

		// Check if sequence is valid and of correct length
		if (!super.isValidSequence(newSequence)) {
			System.out.println("replaceSequenceGenome was called with an invalid sequence. No change.");
			return;
		}
		// Check if the given identifier is present in the alignment
		if (!this.containsIdentifier(identifier)) {
			System.out.println("identifier not present in alignment.");
			return;
		}
		// Get the difference stored in alignment under the given identifier
		String oldDifference = this.getGenome(identifier);
		// Recreate the original genome
		String oldGenome = this.getOriginalGenome(oldDifference);
		// Replace the sequences given to us
		String newGenome = oldGenome.replaceAll(oldSequence, newSequence);
		// Get the difference of this new genome
		String newDifference = this.getGenomeDifference(newGenome);
		// Store result in the alignment
		this.replaceGenome(identifier, newDifference);
	}

	@Override
	/**
	 * See StandardAlignment for details. Since we assume that subString is a sequence of nucleotides 
	 * written in "standard" view, we first have to recreate the standard view of each sequence stored 
	 * in this alignment before comparing their representation with subString.
	 */
	public ArrayList<String> searchGenomes(String subString){
		// Create an empty arraylist to store all ids that have subString in their representation
		ArrayList<String> ids = new ArrayList<String>();

		// Iterate over each (identifier, genome)-pair and look for matches, but first recreate the standard view
		for (String identifier : this.getIdentifiers()) {
			String difference = this.getGenome(identifier);
			String sequence   = this.getOriginalGenome(difference);
			if (sequence.contains(subString)) {
				ids.add(identifier);
			}
		}
		return ids;
	}

	@Override
	/**
	 * See StandardAlignment for details. First, recreate the standard view again, then execute this 
	 * method as specified in the superclass.
	 */
	public void replaceGenome(String oldId, String newId, String newSequence) {
		// Get the difference with the reference genome
		String differenceSequence = getGenomeDifference(newSequence);
		// Add this 'difference genome' to the alignment
		super.replaceGenome(oldId, newId, differenceSequence);
	}

	@Override
	/**
	 * See StandardAlignment for details. We assume that genome is written in the standard view.
	 * Hence, first create the difference view of this genome before storing in the alignment.
	 */
	public void addGenome(String identifier, String genome) {
		// First, get the difference, 
		String difference = this.getGenomeDifference(genome);
		// then, add to alignment
		super.addGenome(identifier, difference);
	}
//	
//	@Override
//	/**
//	 * Our implementation does not allow the reference genome to be deleted. 
//	 */
//	public void removeGenome(String identifier) {
//		// First check if we don't try to remove the reference genome
//		if (identifier == this.getReferenceID()) {
//			System.out.println("Can not remove reference genome of a SNiP alignment.");
//			return;
//		} else {
//			super.removeGenome(identifier);
//		}
//	}

	@Override
	/**
	 * See StandardAlignment. We override this method, as we are going to use the reference genome 
	 * used in constructing the SNiP as the default reference genome used to compute the difference score.
	 */
	public int getDifferenceScore() {
		// Create the difference view of the reference genome of this SNiP alignment.
		// Safer since one is allowed to delete genomes
		String comparisonString = "";
		for (int i = 0; i < this.getLengthGenomes(); i++) {
			comparisonString += '.';
		}
		return super.getDifferenceScore(comparisonString);
	}

	/* Getters and setters */

	public String getReferenceGenome() {
		return this.referenceGenome;
	}

	public String getReferenceID() {
		return referenceID;
	}

	@Override
	/**
	 * See StandardAlignment for details. Override this method as we are copying a SNiPAlignment specifically
	 * instead of just a StandardAlignment object.
	 */
	public SNiPAlignment copy(){
		SNiPAlignment copy = new SNiPAlignment(this.copyAlignment());
		return copy;
	}
}
