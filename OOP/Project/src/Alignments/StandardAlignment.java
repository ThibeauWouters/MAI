package Alignments;

import java.util.*;
import java.io.*;

public class StandardAlignment implements AlignmentInterface{

	/* Variables */
	private ArrayList<String> identifiers = new ArrayList<String>();
	private ArrayList<String> sequences = new ArrayList<String>();
//	private HashMap<String, String> alignment = new HashMap<String, String>();


	/* Constructor */

	// Constructor for this class. fileName should be a FASTA file.
	public StandardAlignment(String fileName) {
		
		// Check if fileName corresponds to a FASTA file. If FASTA file, final 6 characters are ".fasta"
		// TODO can we throw some exception for this?
		String extension = fileName.substring(fileName.length()-6, fileName.length());
		if (!(extension.equals(".fasta"))) {
			System.out.println("Alignment constructor takes .fasta files only. Exiting constructor");
			System.exit(0);
		}
		// TODO: is this ok in terms of getting the filepath etc?
		System.out.println("FASTA file entered: " + fileName); 
		// Scan all lines, add the identifier and genome sequence to respective ArrayLists.
		try(Scanner input = new Scanner(new FileReader(fileName));){ 
			while(input.hasNext()) { 
				try {
					// Extract identifier, but remove ">" at start
					String id = input.next();
					id = id.substring(1);
					this.identifiers.add(id);
					// Extract genome sequence
					String seq = input.next();
					this.sequences.add(seq);
//					this.alignment.put(id, seq);
				} catch (InputMismatchException mismatch) {
					// In case the input is invalid:
					System.out.println("Invalid entry in file found, stop reading input."); 
					break; 
				} 
			}  
		} catch (FileNotFoundException e) {
			// In case the rights to file are invalid:
			// TODO: is this indeed the error in case we don't have the right to access the file?
			System.out.println("Error: file not found or you don't have the rights to access this file?"); 
			System.out.println("Exiting program");
			System.exit(0); 
		} catch (Exception e) {
			// Catch any other exception
			System.out.println("Unexpected error occurred: " + e);
			System.out.println("Exiting program");
			System.exit(0); 
		}
	}

	/* Methods */
	
	@Override
	public String getGenome(String identifier) {
		// Get index of this identifier
		int index = this.identifiers.indexOf(identifier);
		// Get sequence with this index and return
		return this.sequences.get(index);
	}
	
	@Override
	public String getIdentifier(String sequence) {
		// Get index of this sequence
		int index = this.sequences.indexOf(sequence);
		// Get sequence with this index and return
		return this.identifiers.get(index);
	}

	@Override
	public ArrayList<String> searchGenomes(String subString){
		/*
		 * Returns the identifiers of the genomes which contain the given substring 
		 */
		// Older version, without the hashmap
		ArrayList<String> ids = new ArrayList<String>();
		
		// Search all strings
		for (int i = 0; i < this.sequences.size(); i++) {
			String sequence = this.sequences.get(i); 
			if (sequence.contains(subString)) {
				String idName = this.identifiers.get(i);
				ids.add(idName);
			}
		}
		
		return ids;
	}
	
	@Override
	public void replaceGenome(String oldId, String newId, String newSequence) {
		// Get index of identifier and sequence to be deleted
		// TODO: what if index not found, ie, the identifier is not recognized?
		// TODO: what if sequence is not valid? Something different than A, C, T, G
		// TODO: what if sequence is already there?
		int index = this.identifiers.indexOf(oldId);
		
		// Delete them
		this.identifiers.remove(index);
		this.sequences.remove(index);
		
		// Insert the new genome
		this.identifiers.add(index, newId);
		this.sequences.add(index, newSequence);
	}
	
	@Override
	public void replaceSequenceGenome(String identifier, String oldSequence, String newSequence) {
		// Make sure that the lengths of two sequences match, so that length of new genome is preserved
		if(oldSequence.length() != newSequence.length()) {
			System.out.println("replaceSequenceGenome was called with sequences of different length. No change");
			return;
		}
		
		// Get genome
		int index = this.identifiers.indexOf(identifier);
		String oldGenome = this.sequences.get(index);
		// Replace all occurrences, and overwrite the sequence in the database
		String newGenome = oldGenome.replaceAll(oldSequence, newSequence);
		this.sequences.set(index, newGenome);
	}
	
	@Override
	public void replaceSequenceAlignment(String oldSequence, String newSequence) {
		for(String identifier : this.identifiers) {
			this.replaceSequenceGenome(identifier, oldSequence, newSequence);
		}
	}
	
	@Override
	public void addGenome(String identifier, String sequence) {
		this.identifiers.add(identifier);
		this.sequences.add(sequence);
	}
	
	@Override
	public void removeGenome(String identifier) {
		int index = this.identifiers.indexOf(identifier);
		this.identifiers.remove(index);
		this.sequences.remove(index);
	}
	
	
	@Override
	public int getLengthAlignment() {
		return this.identifiers.size();
	}
	
	@Override
	public ArrayList<String> getIdentifiers() {
		return this.identifiers;
	}
	
	@Override
	public ArrayList<String> getSequences() {
		return this.sequences;
	}
	
	/* Main method for testing -- TODO remove after code completion */
	public static void main(String[] args) {
		StandardAlignment sa = new StandardAlignment("hiv.fasta");
		System.out.println(sa.identifiers.get(0));
		
		ArrayList<String> test = sa.searchGenomes("TTT");
		System.out.println(test);
		
		// test of replace genome
		sa.addGenome("newID", "XXX");
		System.out.println(sa.sequences.get(100));
	}
	
	

	/* Getters and setters etc */

	//	public String getIdentifier


}
