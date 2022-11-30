package Alignments;

import java.util.*;
import java.util.Map.Entry;
import java.io.*;

public class StandardAlignment {

	/* Variables */
	// The identifiers and sequences will be stored in a hashmap. The keys are the identifiers,
	// and the values will be the DNA sequences.
	private HashMap<String, String> alignment = new HashMap<String, String>();


	/* Constructor */

	// Note: fileName should be a FASTA file.
	public StandardAlignment(ArrayList<String> ids, ArrayList<String> genomes) {
		// Is there a way to do it in one go?
		for(int i=0;i<ids.size();i++) {
			this.alignment.put(ids.get(i), genomes.get(i));
		}
	}

	/* Methods */

	public String getGenome(String identifier) {
		// Get sequence with this identifier and return
		// TODO - what if sequence is not there? void?
		return this.alignment.get(identifier);
	}

	public String getIdentifier(String sequence) {
		// Get the identifier of the given sequence in the alignment
		// TODO - what to do if this fails? Return nothing?
		if (this.alignment.containsValue(sequence)){
			for(Entry<String, String> entry: this.alignment.entrySet()) {
				// Return the key if it matches the desired sequence
				if(entry.getValue() == sequence) {
					String id = entry.getKey();
					return id;
				}
			}
		} else {
			// TODO - how to handle this?
			System.out.println("Sequence not found.");
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
		// Make sure that the lengths of two sequences match, so that length of new genome is preserved
		if(oldSequence.length() != newSequence.length()) {
			System.out.println("replaceSequenceGenome was called with sequences of different length. No change");
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
		// Replace the sequence for all genomes in the alignment
		this.alignment.forEach((id, genome) -> {
			String newGenome = genome.replaceAll(oldSequence, newSequence);
			// Store in alignment again
			this.alignment.put(id, newGenome);
		});
		
//		for(String identifier : this.alignment.keySet()) {
//			System.out.println(identifier);
//			this.replaceSequenceGenome(identifier, oldSequence, newSequence);
//		}
	}

	public void addGenome(String identifier, String sequence) {
		this.alignment.put(identifier, sequence);
	}

	public void removeGenome(String identifier) {
		this.alignment.remove(identifier);
	}

	public int getSize() {
		return this.alignment.size();
	}

	public ArrayList<String> getIdentifiers() {
		// Get the keys, as keyset, and convert to arraylist
		ArrayList<String> listOfKeys = new ArrayList<String>(this.alignment.keySet());
		return listOfKeys;
	}

	public ArrayList<String> getSequences() {
		ArrayList<String> listOfSequences = new ArrayList<>(this.alignment.values());
		return listOfSequences;
	}

	/* Main method for testing -- TODO remove after code completion */
	public static void main(String[] args) {
		// Read the fasta file
		ReadFasta rf = new ReadFasta("hiv.fasta");
		
		// Store them in the alignment
		StandardAlignment sa = new StandardAlignment(rf.getIds(), rf.getGenomes());
		System.out.println(sa.getIdentifiers());

		System.out.println("Testing search genome");
		ArrayList<String> test = sa.searchGenomes("AAAAAAAAAAAAAAAAAA");
		System.out.println(test);

		System.out.println("Testing add genome");
		sa.addGenome("newID", "XXX");
		System.out.println(sa.getGenome("newID"));
		
		System.out.println("Testing getting genome");
		String test2 = sa.getGenome("2002.A.CD.02.KTB035");
		System.out.println(test2);
		
		System.out.println("Testing getting id");
		String test3 = sa.getIdentifier("XXX");
		System.out.println(test3);
		
		System.out.println("Testing replacing genome");
		sa.replaceGenome("newID", "newerID", "ZZZ");
		System.out.println(sa.getGenome("newerID"));
		
		System.out.println("Testing replaceSequenceGenome");
		sa.replaceSequenceGenome("2002.A.CD.02.KTB035", "A", "XXX");
		System.out.println(sa.getGenome("2002.A.CD.02.KTB035"));
		
		System.out.println("Testing replaceSequenceGenome");
		sa.replaceSequenceGenome("2002.A.CD.02.KTB035", "A", "X");
		System.out.println(sa.getGenome("2002.A.CD.02.KTB035"));
		
		// TODO - this has some weird behaviour?
//		System.out.println("Testing replaceSequence alignment");
//		sa.replaceSequenceAlignment("A", "X");
//		System.out.println(sa.getSequences());
		
		
		System.out.println("Testing remove genome");
		sa.removeGenome("heyheyhey");
		}



	/* Getters and setters etc */

	//	public String getIdentifier


}
