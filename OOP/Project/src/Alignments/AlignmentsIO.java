package alignments;

import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

/**
 * This is a class used to read in the contents of a .fasta file (or .txt file containing the contents of an alignment)
 * and store its contents as instance variables  * such that they can be processed by other classes.
 * All its fields are final, as we are not going to modify the variables stored after a .fasta file has been read.
 * The implemented methods are simply getter functions to access the contents again. We note that since other file
 * extensions are also allowed, we can reuse this class when reading in the backup files made by technical staff. This
 * class also implements static methods which copy an alignment to a file, which is used regularly when writing
 * reports or data to files, or when creating backup files.
 * 
 * @author ThibeauWouters
 */

public class AlignmentsIO {
	/**
	 * ids is an ArrayList storing all the identifiers of a .fasta or .txt file containing an alignment
	 */
	private final ArrayList<String> ids           = new ArrayList<String>();
	/**
	 * genomes is an ArrayList storing all the genome sequences of a .fasta or .txt file containing an alignment
	 */
	private final ArrayList<String> genomes       = new ArrayList<String>();
	/**
	 * fileName holds the name of the file that is read when calling the constructor, in case future
	 * extensions of the application might want to check this name again after a file has been read, which is
	 * useful in case there are several files or if we use different .fasta files as initial input.
	 */
	private final String fileName;

	/**
	 * The constructor takes the filename of a .fasta file (the initial input of the application) or a .txt file
	 * (which is created by employees such as backup alignments). The constructor reads in the contents of the file
	 * and stores the identifiers and genomes as two separate ArrayLists as instance variables.
	 * 
	 * @param fileName Name of the file containing the alignment
	 */
	public AlignmentsIO(String fileName) {
		// Save the filename
		this.fileName = fileName;

		// Open a scanner to read the contents of this file
		try(Scanner input = new Scanner(new FileReader(fileName))){
			while(input.hasNext()) { 
				try {
					// First next line is an id
					String id = input.next();
					// Second next line is a genome
					String seq = input.next();
					// Add them to the respective ArrayLists
					this.ids.add(id);
					this.genomes.add(seq);
				} catch (InputMismatchException mismatch) {
					// In case the input is invalid:
					System.out.println("Invalid entry in file found, stop reading input."); 
					break; 
				} 
			}  
		} catch (FileNotFoundException e) {
			// In case the file is not found
			System.out.println("Error: file not found. Exiting program"); 
			e.printStackTrace();
			System.exit(0); 
		} catch (Exception e) {
			// Catch any other exception here
			System.out.println("Unexpected error occurred. Exiting program.");
			e.printStackTrace();
			System.exit(0); 
		}
	}
	
	/* Methods to write to files */

	/**
	 * Auxiliary, static method that writes down the contents of a StandardAlignment object to a specified file,
	 * using the same format as the .fasta files (that is, printing each identifier followed by its
	 * corresponding genome sequence).
	 * 
	 * @param fileName The name of the file to which we will write the alignment
	 * @param sa The alignment we which to save to the file
	 */
	public static void copyAlignmentToFile(String fileName, StandardAlignment sa) {
		// Save alignment to the file, open a new writer:
		try(BufferedWriter bw = new BufferedWriter(new FileWriter(fileName))){			
			// Iterate over all IDs of our alignment to be saved
			for (String identifier : sa.getIdentifiers()) {
				// Get the corresponding genome with this ID
				String genome = sa.getGenome(identifier);
				// First, write the ID to the file
				bw.write(identifier + "\n");
				// Then, write the genome sequence to the file
				bw.write(genome + "\n");
			}
		}
		catch (FileNotFoundException fnf) {
			// In case file not found
			System.out.println("File not found error occurred while writing alignment.");
			fnf.printStackTrace();
		} 
		catch (IOException ioe) {
			// In case any other IO exception happens
			System.out.println("IOexception occurred while writing alignment.");
			ioe.printStackTrace();
		}	
	} 

	/**
	 * Writes an alignment to a file with a currently open BufferedWriter object which is being used by some other
	 * method to write down an alignment to a file. For instance, when the team leader makes a copy of all alignments
	 * we use this method on each of the Bioinformatician's alignments separately, and save them in one big file,
	 * such that we use an open BufferedWriter object.
	 * 
	 * @param bw An open BufferedWriter which is saving several alignments.
	 * @param sa The current alignment we which to write to the BufferedWriter.
	 * @throws IOException During writing, an IOException can occur.
	 */
	public static void copyAlignmentToFile(BufferedWriter bw, StandardAlignment sa) throws IOException {
		// Write down the alignment, iterate over all IDs
		for (String identifier : sa.getIdentifiers()) {
			// Get the genome of this ID
			String genome = sa.getGenome(identifier);
			// First, write ID
			bw.write(identifier + "\n");
			// Then, write genome
			bw.write(genome + "\n");
		}
	} 
	
	/* Getter functions */
	
	// Note: no setters needed as class is final

	public ArrayList<String> getIdentifiers() {
		return this.ids;
	}

	public ArrayList<String> getGenomes() {
		return this.genomes;
	}

	public String getFileName() {
		return fileName;
	}
}

