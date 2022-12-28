package Alignments;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.*;

public class FastaContents {
	// Create new arraylists to save the ids and genomes into
	private final ArrayList<String> ids     = new ArrayList<String>();
	private final ArrayList<String> genomes = new ArrayList<String>();
	private HashMap<String, String> hmap    = new HashMap<String, String>();

	public String fileName;

	public FastaContents(String fileName) {
		// Save the filename for future convenience
		this.fileName = fileName;
		
		// Check if fileName corresponds to a FASTA file (final 6 characters are ".fasta")
		// TODO can we throw some exception for this?
		String extension = fileName.substring(fileName.length()-6, fileName.length());
		if (!(extension.equals(".fasta"))) {
			System.out.println("Alignment constructor takes .fasta files only. Exiting constructor.");
			System.exit(0);
		}
		// TODO: is this ok in terms of getting the filepath etc?
		System.out.println("FASTA file entered: " + fileName); 
		// Scan all lines, add the identifier and genome sequence to respective ArrayLists.
		try(Scanner input = new Scanner(new FileReader(fileName));){ 
			while(input.hasNext()) { 
				try {
					// Save identifier, but remove ">" at the start
					String id = input.next();
					id = id.substring(1);
					this.ids.add(id);
					// Save genome
					String seq = input.next();
					this.genomes.add(seq);
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
		
		// Also save into the HashMap - this is how our "alignment" looks like
		for(int i=0;i<ids.size();i++) {
			this.hmap.put(ids.get(i), genomes.get(i));
		}
	}
	
	/* Getters */

	public ArrayList<String> getIds() {
		return this.ids;
	}

	public ArrayList<String> getGenomes() {
		return this.genomes;
	}

	public HashMap<String, String> getAlignment() {
		return this.hmap;
	}
	
}

