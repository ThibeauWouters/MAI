package alignments;

import java.util.ArrayList;
import java.util.HashMap;

public class SNiPAlignment extends StandardAlignment {
	
	/* Variables */ 
	// TODO - final? 
	private final String referenceGenome;
	private final String referenceID;

	/*  Constructors */
	
	// Option 1 - create from FASTA file
	
	// TODO - what if reference ID is not there???
	
	
	public SNiPAlignment(FastaContents fasta, String referenceID) {
		// Create SNiP alignment from FASTA file
		
		// TODO - what if we want to CHANGE the reference genome? (Not good to change the whole alignment call new one?)
		// TODO - what if the reference ID is not present in the alignment?
		
		// Create standard alignment first
		super(fasta);
		
		// Get the reference genome from FASTA. Save it to the instance to refer to it later on
		String referenceGenome = super.copyAlignment().get(referenceID);
		this.referenceGenome = referenceGenome;
		// Also save the ID we used, to find it again later on:
		this.referenceID = referenceID;
		
		// Edit from standard alignment to SNiP, get "difference" of all sequences
		this.replaceDifferences();
	}
	
	// Same as above, but with default for referenceID

	public SNiPAlignment(FastaContents fasta) {		
		this(fasta, fasta.getIdentifiers().get(0));
	}
	
	public SNiPAlignment(HashMap<String, String> hmap, String referenceID) {
		// TODO - what if we want to CHANGE the reference genome? (Not good to change the whole alignment call new one?)
		// TODO - what if the reference ID is not present in the alignment?
		
		// Create standard alignment first
		super(hmap);
		
		// Get the reference genome from HashMap. Save it to the instance to refer to it later on
		this.referenceID = referenceID;
		String referenceGenome = super.copyAlignment().get(referenceID);
		this.referenceGenome = referenceGenome;
		
		// Edit from standard alignment to SNiP, get "difference" of all sequences
		this.replaceDifferences();
	}
	
	public SNiPAlignment(HashMap<String, String> hmap) {
		// Same as above, but default reference genome to first in list
		
		this(hmap, hmap.keySet().iterator().next());
	}
	
	public SNiPAlignment(StandardAlignment sa, String referenceID) {
		// Create standard alignment first
		super(sa);
		
		// Get the reference genome from FASTA. Save it to the instance to refer to it later on
		this.referenceID = referenceID;
		String referenceGenome = sa.getGenome(referenceID);
		this.referenceGenome = referenceGenome;
		
		// Edit from standard alignment to SNiP, get "difference" of all sequences
		this.replaceDifferences();
	}
	
	public SNiPAlignment(StandardAlignment sa) {
		// Same as above, but use the first ID as reference by default
		
		this(sa, sa.getIdentifiers().get(0));
	}
	
	public SNiPAlignment(SNiPAlignment snip) {
		// Create alignment in super first
		//super(snip);
		
		// Overwrite with a copy from a given, existing SNiPAlignment
		this.setAlignment(snip.copyAlignment());
		this.referenceGenome = snip.getReferenceGenome();
		this.referenceID     = snip.getReferenceID();
	}
	

	/* Methods */
	
	public static String getGenomeDifference(String referenceGenome, String genome) {
		// Edits a single genome (in standard alignment) and gets the differences in the sequences compared to a reference genome for SNiP alignment.
		
		// TODO - check if equal length
		
		// Construct a string containing the differences
		String differences = "";
		// Iterate over all characters. If identical, add dot, otherwise add character of "genome" string
		for(int i = 0; i<genome.length(); i++) {
			if(referenceGenome.charAt(i) == genome.charAt(i)) {
				differences += ".";
			} else {
				differences += genome.charAt(i);
			}
		}
		return differences;
	}
	
	public String getGenomeDifference(String genome) {
		/*
		 * Same as above, but non-static version.
		 */
		
		return getGenomeDifference(this.referenceGenome, genome);
	}
	
	public void replaceDifferences() {
		// Iterates over all (id, seq)-pairs in the alignment, gets the difference, and save with that ID
		
		for (String id : this.getIdentifiers()) {
			String genome = this.getGenome(id);
			String difference = this.getGenomeDifference(genome);
			this.replaceGenome(id, difference);
		}
	}
	
	public static String getOriginalGenome(String referenceGenome, String difference) {
		/*
		 * Given a genome with dots (differences), recreates the original genome by using the reference genome.
		 */
		
		// Construct a string containing the differences
		String original = "";
		// Iterate over all characters. If a dot, add char of reference genome, otherwise add char "difference" string
		for (int i = 0; i<referenceGenome.length(); i++) {
			if(difference.charAt(i) == '.') {
				original += referenceGenome.charAt(i);
			} else {
				original += difference.charAt(i);
			}
		}
		return original;
	}
	
	public String getOriginalGenome(String difference) {
		/*
		 * Same as above, but non-static version.
		 */
		
		return getOriginalGenome(this.referenceGenome, difference);
	}
	
	@Override
	public void replaceSequenceGenome(String identifier, String oldSequence, String newSequence) {

		// Check if sequence is valid and of correct length
		if (!super.isValidSequence(newSequence)) {
			System.out.println("replaceSequenceGenome was called with sequences of different length. No change");
			return;
		}
		
		// Find the corresponding 'difference' genome
		String oldDifference = this.getGenome(identifier);
		// Recreate the original genome
		String oldGenome = this.getOriginalGenome(oldDifference);

		// Modify similarly as in standard version
		String newGenome = oldGenome.replaceAll(oldSequence, newSequence);
		// Get the difference of this new genome
		String newDifference = this.getGenomeDifference(newGenome);
		// Store in the alignment again
		this.addGenome(identifier, newDifference);
	}
	
	@Override
	public ArrayList<String> searchGenomes(String subString){
		// Returns the identifiers of the genomes which contain the given substring
		
		// Create an empty arraylist of all ids
		ArrayList<String> ids = new ArrayList<String>();

		// Iterate over each (identifier, genome)-pair and look for matches. 
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
	public void replaceGenome(String oldId, String newId, String newSequence) {
		// Get the difference with the reference genome
		String differenceSequence = getGenomeDifference(this.referenceGenome, newSequence);
		// Add this 'difference genome' to the alignment
		super.replaceGenome(oldId, newId, differenceSequence);
	}

	@Override
	public void addGenome(String identifier, String sequence) {
		// First, get the difference, 
		String difference = this.getGenomeDifference(sequence);
		// then, add to alignment
		super.addGenome(identifier, difference);
	}
	
	@Override
	public int getDifferenceScore() {
		// We override this method, as we are going to use the reference genome used in constructing the SNiP to compute the difference score.
		
		return super.getDifferenceScore(this.getGenome(this.getReferenceID()));
	}
	
	/* New getters and setters */
	
	public String getReferenceGenome() {
		return this.referenceGenome;
	}

	public String getReferenceID() {
		return referenceID;
	}
	
	@Override
	public SNiPAlignment copy(){
		// Returns a deep copy of this alignment as hashmap. Otherwise, Objects using another alignment will edit that alignment
		
		SNiPAlignment copy = new SNiPAlignment(this.copyAlignment());
		return copy;
	}

}
