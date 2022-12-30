package alignments;

import java.util.*;
import java.util.Map.Entry;

public class StandardAlignment {

	/* Variables */
	// The identifiers and sequences will be stored in a hashmap. The keys are the identifiers, the values will be the DNA sequences.
	private HashMap<String, String> alignment      = new HashMap<String, String>();
	public final static Set<Character> NUCLEOTIDES   = new HashSet<>(Arrays.asList('A', 'C', 'T', 'G', '.')); 


	/* Constructor */

	public StandardAlignment(FastaContents fasta) {
		// Load the contents of the fasta file into the alignment:
		this.setAlignment(fasta.getAlignment());
	}
	
	public StandardAlignment(HashMap<String, String> hmap) {
		// Load the contents of the fasta file into the alignment:
		this.setAlignment(hmap);
	}
	
	public StandardAlignment(StandardAlignment sa) {
		// Load the contents of a different standard alignment into the alignment:
		
		// Copy the alignment
		this.setAlignment(sa.copyAlignment());
	}
	
	public StandardAlignment(SNiPAlignment snip) {
		// For each ID in the SNIP, recreate the original genome before storing
		
		// Copy the alignment
		this.setAlignment(snip.copyAlignment());
		
		// Then edit back such that we have genome rather than difference
		for (String id : snip.getIdentifiers()) {
			String difference = snip.getGenome(id);
			String genome     = snip.getOriginalGenome(difference);
			this.replaceGenome(id, genome);
		}
	}
	
	// Empty constructor
	public StandardAlignment() {};

	/* Methods */

	public String getGenome(String identifier) {
		// Get sequence with this identifier and return
		// Get the identifier of the given sequence in the alignment
		// TODO - what to do if this fails? Return -1? Exit entirely?

		if (this.getIdentifiers().contains(identifier)){
			return this.alignment.get(identifier);
		} else {
			// TODO - how to handle this?
			System.out.println("Sequence not found.");
		}
		return "-1";
	}

	public String getIdentifier(String sequence) {
		// Get the identifier of the given sequence in the alignment
		// TODO - what to do if this fails? Return -1? Exit entirely?
		if (this.getGenomes().contains(sequence)){
			for(Entry<String, String> entry: this.alignment.entrySet()) {
				// Return the key if it matches the desired sequence
				if(entry.getValue().equals(sequence)) {
					return entry.getKey();
				}
			}
		} else {
			// TODO - how to handle this?
			System.out.println("Identifier not found.");
		}
		return "-1";
	}

	public ArrayList<String> searchGenomes(String subString){
		/*
		 * Returns the identifiers of the genomes which contain the given substring 
		 */

		// Create an empty arraylist of all ids
		ArrayList<String> ids = new ArrayList<String>();

		// Iterate over each (identifier, genome)-pair, look for matches. 
		this.copyAlignment().forEach((id, seq) -> {
			if (seq.contains(subString)) {
				ids.add(id);
			}
		});
		return ids;
	}
	
	public void addGenome(String identifier, String genome) {
		if (isValidGenome(genome)) {
			this.alignment.put(identifier, genome);
		} else {
			System.out.println("addGenome is called with an invalid sequence. No change.");
		}
	}

	public void removeGenome(String identifier) {
		this.alignment.remove(identifier);
	}

	public void replaceGenome(String oldId, String newId, String newGenome) {
		// Get index of identifier and sequence to be deleted
		
		// TODO: is this OK? What if we want to give a sequence rather than ID?
		// TODO: for such functions and other similar ones: check if sequences are valid?
		// TODO: what if index not found, ie, the identifier is not recognized?
		// TODO: what if sequence is not valid? Something different than A, C, T, G
		// TODO: what if sequence is already there?

		if (isValidGenome(newGenome)) {
			this.alignment.remove(oldId);
			this.alignment.put(newId, newGenome);
		} else {
			System.out.println("replaceGenome is called with an invalid sequence. No change.");
		}
	}
	
	public void replaceGenome(String oldId, String newSequence) {
		this.replaceGenome(oldId, oldId, newSequence);
	}

	public void replaceSequenceGenome(String identifier, String oldSequence, String newSequence) {

		// Check if the provided sequence is valid, otherwise don't change.
		if (!isValidSequence(newSequence)){
			System.out.println("replaceSequenceGenome was called with an invalid sequence. No change.");
			return;
		}

		// Get the original genome and do the replacement
		String oldGenome = this.getGenome(identifier);
		String newGenome = oldGenome.replaceAll(oldSequence, newSequence);
		// Store in alignment again
		this.addGenome(identifier, newGenome);
	}

	public void replaceSequenceAlignment(String oldSequence, String newSequence) {

		// Check if the provided sequence is valid, otherwise don't change.
		if (!isValidSequence(newSequence)){
			System.out.println("replaceSequenceGenome was called with an invalid sequence. No change.");
			return;
		}

		// If valid, then change it everywhere
		for (String id: this.getIdentifiers()) {
			this.replaceSequenceGenome(id, oldSequence, newSequence);
		}
	}

	/* Getters and setters */

	public int getSize() {
		return this.alignment.size();
	}
	
	public int getLengthGenomes() {
		return this.getGenomes().get(0).length();
	}

	public ArrayList<String> getIdentifiers() {
		// Get the keys, as keyset, and convert to arraylist
		ArrayList<String> listOfKeys = new ArrayList<String>(this.alignment.keySet());
		return listOfKeys;
	}

	public ArrayList<String> getGenomes() {
		ArrayList<String> listOfSequences = new ArrayList<>(this.alignment.values());
		return listOfSequences;
	}

	public HashMap<String, String> copyAlignment(){
		// Returns a deep copy of this alignment as hashmap. Otherwise, Objects using another alignment will edit that alignment
		
		HashMap<String, String> copy = new HashMap<String, String>();
        for (Map.Entry<String, String> entry : this.alignment.entrySet()) {
        	copy.put(entry.getKey(), entry.getValue());
        }
		return copy;
	}
	
	public StandardAlignment copy(){
		// Returns a deep copy of this alignment as hashmap. Otherwise, Objects using another alignment will edit that alignment
		
		StandardAlignment copy = new StandardAlignment(this.copyAlignment());
		return copy;
	}
	
	public void setAlignment(HashMap<String, String> alignment){
		this.alignment = alignment;
	}

	/* Methods to compute the difference score */

	public static int computeDifferenceScorePair(String firstGenome, String secondGenome) {
		// Computes the difference between two genomes, i.e., the number of different positions/characters between them.

		int score = 0;

		// Iterate over all characters of both genomes. If different, we add 1
		for(int i = 0; i<firstGenome.length(); i++) {
			if(firstGenome.charAt(i) != secondGenome.charAt(i)) {
				score += 1;
			}
		}
		return score;
	}

	
	public int getDifferenceScore(String referenceGenome) {
		// Computes the difference score of this alignment.
		
		int score = 0;
		// Iterate over all the genomes in this list
		for (String genome : this.getGenomes()) {
			score += computeDifferenceScorePair(referenceGenome, genome);
		}
		return score;	
	}
	

	public int getDifferenceScore() {
		// Same as above, but uses the first genome by default.
		
		ArrayList<String> allGenomes = this.getGenomes();
		String referenceGenome = allGenomes.get(0);
		return getDifferenceScore(referenceGenome);
	}
	
	public static boolean isValidSequence(String sequence) {
		// Get all characters as an array
		char[] allChars = sequence.toCharArray();

		// Convert them to a set
		Set<Character> setOfChars = new HashSet<>();
		for (char c : allChars) {
			setOfChars.add(c);
		}

		return NUCLEOTIDES.containsAll(setOfChars);
	}
	
	
	public boolean isValidGenome(String genome) {
		// Checks whether or not a provided sequence is valid for a genome (i.e., correct length and only contains A, C, T, G)
		
		// 1) Check the length (assume all genomes in alignment are valid, i.e. have same length)
		int genomeLength = this.getLengthGenomes();
		if (genome.length() != genomeLength) {
			return false;
		}
		
		// 2) Check if it's a correct sequence
		return isValidSequence(genome);
	}
	
	public void display() {
		// Displays the initial few genomes and part of their nucleotides for visualization properties
		
		
		// Display the first three ids and genomes as illustration
		int numberToShow = Math.min(this.getSize(), 3);
		
		// Get the number of nucleotides to show in each of those genomes
		int numberOfNucleotides = Math.min(this.getLengthGenomes(), 70);
		
		// Display them:
		for (int i = 0; i<numberToShow;i++) {
			String id = this.getIdentifiers().get(i);
			System.out.println(id);
			System.out.println(this.getGenome(id).substring(0, numberOfNucleotides) + " (cont.)");
		}
	}
}
