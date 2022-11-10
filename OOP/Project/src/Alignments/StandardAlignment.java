package Alignments;

import java.util.*;
import java.io.*;

public class StandardAlignment implements AlignmentInterface{

	/* Variables */
	private ArrayList<String> identifiers = new ArrayList<String>();
	private ArrayList<String> sequences = new ArrayList<String>();


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
	public ArrayList<String> searchGenomes(String subString){
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
		// TODO: implement this
	}
	
	/* Main method for testing -- TODO remove after code completion */
	public static void main(String[] args) {
		StandardAlignment sa = new StandardAlignment("hiv.fasta");
		System.out.println(sa.identifiers.get(0).toString());
		
		ArrayList<String> test = sa.searchGenomes("TTT");
		System.out.println(test);
	}
	
	

	/* Getters and setters etc */

	//	public String getIdentifier


}
