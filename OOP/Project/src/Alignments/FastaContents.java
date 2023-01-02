package alignments;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.*;


/**
 * This is a class used to read in the contents of a .fasta file and store it as instance variables
 * such that the contents can be processed by other classes. All its fields are finals, as we are not 
 * going to modify the variables stored there after a .fasta file has been read. We note that other file
 * extensions, such as .txt, are also allowed, and hence we reuse this class when reading in backup files. 
 * 
 * @author ThibeauWouters
 */

public class FastaContents {
	/**
	 * ids is an ArrayList storing all the identifiers of the .fasta file
	 */
	private final ArrayList<String> ids           = new ArrayList<String>();
	/**
	 * genomes is an ArrayList storing all the genome sequences of the .fasta file
	 */
	private final ArrayList<String> genomes       = new ArrayList<String>();
	/**
	 * fileName holds the name of the .fasta file that we are going to read, in case future
	 * extensions of the application might want to check this after a .fasta file has been read
	 * in case there are multiple .fasta files
	 */
	private final String fileName;

	/**
	 * The constructor takes the filename (which we create in the main program) pointing towards a .fasta file
	 * which we are going to read in.
	 * 
	 * @param fileName Name of the file containing the alignment we are going to read in the constructor
	 */
	public FastaContents(String fileName) {
		// Save the filename for future convenience
		this.fileName = fileName;

		try(Scanner input = new Scanner(new FileReader(fileName));){ 
			while(input.hasNext()) { 
				try {
					String id = input.next();
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
			System.out.println("Error: file not found. Exiting program"); 
			e.printStackTrace();
			System.exit(0); 
		} catch (Exception e) {
			// Catch any other exception
			System.out.println("Unexpected error occurred. Exiting program.");
			e.printStackTrace();
			System.exit(0); 
		}
	}
	
	/* Getter functions */

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

