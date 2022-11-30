package Alignments;

import java.util.*;


public class SNiPAlignment extends StandardAlignment {
	private HashMap<String, String> alignment = new HashMap<String, String>();
	private String referenceGenome;

	/*  Constructor */
	// Reference ID is the identifier for the genome that is the reference sequence for constructing this alignment
	public SNiPAlignment(ArrayList<String> ids, ArrayList<String> genomes, String referenceID) {
		// TODO - what if we want to CHANGE the reference genome? (Not good to change the whole alignment call new one?)
		// Call super constructor
//		super(ids, genomes);

		// Then, edit the alignment to get the differences

		// Create a new hashmap where we are going to store the differences in genomes
		HashMap<String, String> newAlignment = new HashMap<String, String>();
		// Get the reference genome. Save it to the instance to refer to it later on
		String referenceGenome = this.alignment.get(referenceID);
		this.referenceGenome = referenceGenome;

		// Iterate over each (identifier, genome)-pair, and get difference. 
		this.alignment.forEach((id, genome) -> {
			String difference = getDifference(referenceGenome, genome);
			System.out.println(difference);
			newAlignment.put(id, difference);
		});
		// Save the new alignment into the corresponding field
		this.alignment = newAlignment;
	}


	public SNiPAlignment(ArrayList<String> ids, ArrayList<String> genomes) {
		// Second constructor: if no reference ID was specified, take first one.
		this(ids, genomes, ids.get(0));
	}

	public String getReferenceGenome() {
		return this.referenceGenome;
	}


	protected static String getDifference(String referenceGenome, String genome) {
		/*
		 * Edits the genomes of a standard alignment and saves only the differences in the sequences.
		 */
		// Construct a string containing the differences
		String differences = "";
		// Iterate over all characters. If identical, add dot, otherwise add character of "genome" string
		for(int i = 0; i<referenceGenome.length(); i++) {
			if(referenceGenome.charAt(i) == genome.charAt(i)) {
				differences += ".";
			} else {
				differences += genome.charAt(i);
			}
		}
		return differences;
	}

	public static void main(String[] args) {
		//		System.out.println("testing difference");
		//		String test = SNiPAlignment.getDifference("ABA", "ACA");
		//		System.out.println(test);

		// Read the fasta file
		ReadFasta rf = new ReadFasta("hiv.fasta");

		// Store them in the alignment
		SNiPAlignment snip = new SNiPAlignment(rf.getIds(), rf.getGenomes());
		System.out.println(snip.getIdentifiers());
		System.out.println(snip.getSequences());

	
		//		System.out.println(snip.getSequences().get(0));
		//		System.out.println(snip.getSequences().get(3));
		System.out.println(snip.getReferenceGenome());
	}
}
