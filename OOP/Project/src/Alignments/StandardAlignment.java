package Alignments;

import java.util.*;
import java.util.Map.Entry;
//import java.io.*;

public class StandardAlignment {

	/* Variables */
	// The identifiers and sequences will be stored in a hashmap. The keys are the identifiers, the values will be the DNA sequences.
	protected HashMap<String, String> alignment    = new HashMap<String, String>();
	public final static Set<Character> NUCLEOTIDES = new HashSet<>(Arrays.asList('A', 'C', 'T', 'G', '.')); 


	/* Constructor */

	public StandardAlignment(FastaContents fasta) {
		// Load the contents of the fasta file into the alignment:
		this.alignment = fasta.getAlignment();
	}

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
		this.alignment.forEach((id, seq) -> {
			if (seq.contains(subString)) {
				ids.add(id);
			}
		});
		return ids;
	}

	public void replaceGenome(String oldId, String newId, String newSequence) {
		// Get index of identifier and sequence to be deleted
		// TODO: is this OK? What if we want to give a sequence rather than ID?
		// TODO: for such functions and other similar ones: check if sequences are valid?
		// TODO: what if index not found, ie, the identifier is not recognized?
		// TODO: what if sequence is not valid? Something different than A, C, T, G
		// TODO: what if sequence is already there?

		this.alignment.remove(oldId);
		this.alignment.put(newId, newSequence);
	}

	public void replaceSequenceGenome(String identifier, String oldSequence, String newSequence) {

		// Check if the provided sequence is valid, otherwise don't change.
		if (!isValidSequence(oldSequence, newSequence)){
			System.out.println("replaceSequenceGenome was called with an invalid sequence. No change.");
			return;
		}

		// Get the original genome and do the replacement
		String oldGenome = this.alignment.get(identifier);
		String newGenome = oldGenome.replaceAll(oldSequence, newSequence);
		// Store in alignment again
		this.alignment.put(identifier, newGenome);
	}

	public void replaceSequenceAlignment(String oldSequence, String newSequence) {
		// TODO - weird behaviour

		/*
		 * Call replaceSequenceGenome on each genome in the alignment:
		 */

		// Check if the provided sequence is valid, otherwise don't change.
		if (!isValidSequence(oldSequence, newSequence)){
			System.out.println("replaceSequenceGenome was called with an invalid sequence. No change.");
			return;
		}

		// If valid, then change it everywhere
		for (String id: this.getIdentifiers()) {
			this.replaceSequenceGenome(id, oldSequence, newSequence);
		}
	}

	public void addGenome(String identifier, String sequence) {
		this.alignment.put(identifier, sequence);
	}

	public void removeGenome(String identifier) {
		this.alignment.remove(identifier);
	}

	public static boolean isValidSequence(String sequence) {
		/*
		 * Checks whether or not a provided sequence is valid for a genome (i.e., only contains A, C, T, G)
		 */

		// Get all characters as an array
		char[] allChars = sequence.toCharArray();

		// Convert them to a set
		Set<Character> setOfChars = new HashSet<>();
		for (char c : allChars) {
			setOfChars.add(c);
		}

		return NUCLEOTIDES.containsAll(setOfChars);
	}

	public static boolean isValidSequence(String oldSequence, String newSequence) {
		/*
		 * If two arguments are provided, check if length is preserved, then check if characters are valid.
		 */

		if(oldSequence.length() != newSequence.length()) {
			System.out.println("Different length");
			return false;
		} else {
			return isValidSequence(newSequence);
		}
	}


	/* Getters and setters */

	public int getSize() {
		return this.alignment.size();
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

	public HashMap<String, String> getAlignment(){
		return this.alignment;
	}

	/* Methods to compute the difference score */

	public static int computeDifferenceScorePair(String firstGenome, String secondGenome) {
		/*
		 * Computes the difference between two genomes, i.e., the number of different positions/characters between them.
		 */

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
		/*
		 * Computes the difference score of this alignment. Saves it as instance variable.
		 */

		int score = 0;

		// Iterate over all the genomes in this list
		for (String genome : this.getGenomes()) {
			score += computeDifferenceScorePair(referenceGenome, genome);
		}
		
		return score;	
	}
	

	public int getDifferenceScore() {
		/*
		 * Same as above, but uses the first genome by default.
		 */
		
		ArrayList<String> allGenomes = this.getGenomes();
		String referenceGenome = allGenomes.get(0);
		return getDifferenceScore(referenceGenome);
	}


}
