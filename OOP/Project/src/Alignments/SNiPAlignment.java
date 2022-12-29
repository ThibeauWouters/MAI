package Alignments;

import java.util.ArrayList;
import java.util.HashMap;

public class SNiPAlignment extends StandardAlignment {
	
	/* Variables */ 
	private HashMap<String, String> alignment = new HashMap<String, String>();
	private String referenceGenome;
	private String referenceID;

	/*  Constructors */
	
	// Reference ID is the identifier for the genome that is the reference sequence for constructing this alignment
	
	// Option 1 - create from FASTA file
	public SNiPAlignment(FastaContents fasta, String referenceID) {
		// TODO - what if we want to CHANGE the reference genome? (Not good to change the whole alignment call new one?)
		// TODO - what if the reference ID is not present in the alignment?
		
		// Create standard alignment first
		super(fasta);
		
		// Get the reference genome from FASTA. Save it to the instance to refer to it later on
		String referenceGenome = super.getAlignment().get(referenceID);
		this.referenceGenome = referenceGenome;
		// Also save the ID we used, to find it again later on:
		this.referenceID = referenceID;
		
		// Edit the entries to get differences rather than 
		for (String id : this.getIdentifiers()) {
			String genome = this.getGenome(id);
			String difference = this.getGenomeDifference(genome);
			this.replaceGenome(id, difference);
		}
	}
	
	// Same as above, but with default for referenceID

	public SNiPAlignment(FastaContents fasta) {		
		this(fasta, fasta.getIdentifiers().get(0));
	}
	
	// Option 2 - create from given HashMap
	public SNiPAlignment(HashMap<String, String> hmap, String referenceID) {
		// TODO - what if we want to CHANGE the reference genome? (Not good to change the whole alignment call new one?)
		// TODO - what if the reference ID is not present in the alignment?
		
		// Create standard alignment first
		super(hmap);
		
		// Get the reference genome from FASTA. Save it to the instance to refer to it later on
		String referenceGenome = super.getAlignment().get(referenceID);
		this.referenceGenome = referenceGenome;
		// Also save the ID we used, to find it again later on:
		this.referenceID = referenceID;
		
		// Edit the entries to get differences rather than 
		for (String id : this.getIdentifiers()) {
			String genome = this.getGenome(id);
			String difference = this.getGenomeDifference(genome);
			this.replaceGenome(id, difference);
		}
	}
	
	public SNiPAlignment(HashMap<String, String> hmap) {		
		this(hmap, hmap.keySet().iterator().next());
	}
	
	
	// Second option: build from a given standard alignment
	
//	public SNiPAlignment(StandardAlignment sa, String referenceID) {
//		// Create standard alignment first
//		super(sa.getAlignment());
//		
//		// Get the reference genome from FASTA. Save it to the instance to refer to it later on
//		String referenceGenome = sa.getGenome(referenceID);
//		this.referenceGenome = referenceGenome;
//		// Also save the ID we used, to find it again later on:
//		this.referenceID = referenceID;
//		
//		// Construct the alignment
//		for (String id : sa.getIdentifiers()) {
//			String genome = sa.getGenome(id);
//			this.replaceGenome(id, genome);
//		}
//	}
	
//	public SNiPAlignment(StandardAlignment sa) {
//		// Same as above, but use the first ID as reference by default
//		this(sa, sa.getIdentifiers().get(0));
//	}
	
//	public SNiPAlignment(SNiPAlignment snip) {
//		
//		super(snip);
//		
//		// Construct a copy from a given, existing SNiPAlignment
//		this.referenceGenome = snip.getReferenceGenome();
//		this.referenceID     = snip.getReferenceID();
//		this.alignment       = snip.getAlignment();
//	}
	

	/* Methods */
	
	public static String getGenomeDifference(String referenceGenome, String genome) {
		/*
		 * Edits a single genome (in standard alignment) and gets the differences in the sequences compared to a reference genome for SNiP alignment.
		 */
		
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
		if (!super.isValidSequence(oldSequence, newSequence)) {
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
		this.alignment.put(identifier, newDifference);
	}
	
	@Override
	public ArrayList<String> searchGenomes(String subString){
		/*
		 * Returns the identifiers of the genomes which contain the given substring 
		 */
		
		// Create an empty arraylist of all ids
		ArrayList<String> ids = new ArrayList<String>();

		// Iterate over each (identifier, genome)-pair, look for matches. 
		this.alignment.forEach((id, difference) -> {
			String seq = this.getOriginalGenome(difference);
			if (seq.contains(subString)) {
				ids.add(id);
			}
		});
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
		alignment.put(identifier, difference);
	}
	
	@Override
	public int getDifferenceScore() {
		/*
		 * We override, as we are going to use the reference genome used in constructing the SNiP to compute the difference score.
		 */
		
		return super.getDifferenceScore(this.getGenome(this.getReferenceID()));
	}
	
	/* Getters and setters */
	
	public String getReferenceGenome() {
		return this.referenceGenome;
	}

	public String getReferenceID() {
		return referenceID;
	}

}
