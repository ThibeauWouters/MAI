package alignments;

import java.util.*;
import java.util.Map.Entry;

/**
 * This is a class that implements all functionalities related to the standard alignment.
 * The alignment is stored as a Java Hashmap, with the identifiers stored as keys, and the corresponding values
 * being the genome sequences provided in a .fasta file. Operations such as editing sequences and adding/removing
 * sequences are implemented here as well.
 * 
 * @author ThibeauWouters
 *
 */
public class StandardAlignment {

	/* Variables */

	/**
	 * Alignment is a HashMap storing the identifiers as keys and the genome sequences as values.
	 */
	private HashMap<String, String> alignment      = new HashMap<String, String>();
	/**
	 * Nucleotides is a constant, and represents a set of all characters which are allowed to appear
	 * in a genome sequence, namely the nucleotides and a dot which is used by SNiPAlignments..
	 */
	public final static Set<Character> NUCLEOTIDES = new HashSet<>(Arrays.asList('A', 'C', 'T', 'G', '.')); 


	/* Constructor */

	/**
	 * Constructor which takes the contents of a Fasta file, as stored by the FastaContents class, and copies
	 * the alignment stored in that Object to the alignment of this standard alignment.
	 * 
	 * @param fasta FastaContents object holding the alignment read from a .fasta file.
	 */
	public StandardAlignment(FastaContents fasta) {
		// Load the contents of the fasta file into the alignment:
		if (!(fasta == null)) {
			for(int i=0; i<fasta.getIdentifiers().size(); i++) {
				this.alignment.put(fasta.getIdentifiers().get(i), fasta.getGenomes().get(i));
			}
		}
	}

	/**
	 * Constructor taking a HashMap, storing an alignment, and copying it to the alignment field of this object.
	 * 
	 * @param hmap HashMap of identifier and sequence pairs.
	 */
	public StandardAlignment(HashMap<String, String> hmap) {
		// Load the contents of the fasta file into the alignment:
		if (!(hmap == null)) {
			for (String id : hmap.keySet()) {
				String genome = hmap.get(id);
				this.alignment.put(id, genome);
			}
		}
	}

	/**
	 * Constructor taking another StandardAlignment instance, and copying its alignment to the alignment field 
	 * to create a new StandardAlignment instance.
	 * 
	 * @param sa A StandardAlignment object.
	 */
	public StandardAlignment(StandardAlignment sa) {
		// Load the contents of a different standard alignment into the alignment:

		if (!(sa == null)) {
			this.alignment = sa.copyAlignment();
		}
	}

	/**
	 * Constructor taking a SNiPAlignment object and copying its alignment to the alignment of this new instance, 
	 * after editing it such that it again represents genome sequences instead of differences between sequences.
	 * 
	 * @param snip A SNiPAlignment object
	 */
	public StandardAlignment(SNiPAlignment snip) {
		// For each ID in the SNIP, recreate the original genome before storing

		// Copy the alignment
		if (!(snip == null)) {
			this.alignment = snip.copyAlignment();

			// Then edit back such that we have genome rather than difference
			for (String id : snip.getIdentifiers()) {
				String difference = snip.getGenome(id);
				String genome     = snip.getOriginalGenome(difference);
				this.alignment.replace(id, genome);
			}
		}
	}

	/**
	 * Empty constructor, used for constructors of the subclass SNiPAlignment.
	 */
	public StandardAlignment() {};

	/* Methods */

	/**
	 * Searches through all the genomes of the alignment and returning those sequences 
	 * which contain a given substring as part of their genome.
	 * 
	 * @param subString Sequence with which we will look for matches within the genome sequences
	 * @return ArrayList storing the identifiers of the genome sequences which contain the specified
	 * string as a substring.
	 */
	public ArrayList<String> searchGenomes(String subString){
		// Create an empty arraylist to store the ids
		ArrayList<String> ids = new ArrayList<String>();

		// Iterate over each (identifier, genome)-pair, look for matches between strings. 
		this.copyAlignment().forEach((id, seq) -> {
			if (seq.contains(subString)) {
				ids.add(id);
			}
		});
		return ids;
	}

	/**
	 * Adds a new (identifier, genome)-pair into the alignment. Checks whether the given
	 * genome has a valid representation.
	 * 
	 * @param identifier The ID of the genome to be added.
	 * @param genome The genome sequence to be added.
	 */
	public void addGenome(String identifier, String genome) {
		// Check if the given genome sequence is valid and the ID is not in use already
		if (!(this.isValidGenome(genome))) {
			System.out.println("addGenome is called with an invalid sequence. No change.");
			return;
		} else if (this.containsIdentifier(identifier)) {
			System.out.println("ID already present in alignment. No change.");
			return;
		} else {
			// Store the new (ID, sequence)-pair
			this.alignment.put(identifier, genome);
		}
	}

	/**
	 * Deletes the (identifier, genome)-pair with specified identifier from the alignment.
	 * 
	 * @param identifier The ID of the genome which we wish to delete.
	 */
	public void removeGenome(String identifier) {
		if (this.containsIdentifier(identifier)) {
			this.alignment.remove(identifier);
		} else {
			System.out.println("removeGenome called with ID not present in alignment.");
		}
	}

	/**
	 * Replaces an existing (identifier, sequence)-pair with a new pair.
	 * 
	 * @param oldId The ID of the existing genome sequence we wish to replace
	 * @param newId The ID of the new genome sequence we wish to add
	 * @param newGenome The new genome sequence to be added to the alignment
	 */
	public void replaceGenome(String oldId, String newId, String newGenome) {
		// Check of the given genome is valid
		if (!this.isValidGenome(newGenome)) {
			System.out.println("replaceGenome is called with an invalid sequence.");
			return;
		// Check if oldId is present in the alignment
		} else if (!(this.containsIdentifier(oldId))) {
			System.out.println("replaceGenome does not recognize oldId. Use addGenome instead.");
		} else {
			// Remove the old pair
			this.alignment.remove(oldId);
			// Add a new pair
			this.alignment.put(newId, newGenome);
		}
	}

	/**
	 * Replaces an existing (identifier, sequence)-pair in the alignment, but reuses the old ID.
	 * 
	 * @param oldId The ID of the genome sequence we wish to replace 
	 * @param newSequence The new genome sequence to be added.
	 */
	public void replaceGenome(String oldId, String newSequence) {
		this.replaceGenome(oldId, oldId, newSequence);
	}

	/**
	 * Replaces a particular sequence of nucleotides of a given identifier specifying a genome
	 * with another sequence of nucleotide characters.
	 * 
	 * @param identifier The ID of the genome in which we want to replace characters.
	 * @param oldSequence The old sequence of characters which has to be replaced.
	 * @param newSequence The new sequence of characters which has to replace the old one.
	 */
	public void replaceSequenceGenome(String identifier, String oldSequence, String newSequence) {
		// First, check if the provided sequence is valid
		if (isValidSequence(newSequence) && oldSequence.length() == newSequence.length()){
			// Get the original genome and do the replacement in that String
			String oldGenome = this.getGenome(identifier);
			String newGenome = oldGenome.replaceAll(oldSequence, newSequence);
			// Store in alignment again
			this.replaceGenome(identifier, newGenome);
		} else {
			System.out.println("replaceSequenceGenome was called with an invalid sequence. No change.");
		}
	}

	/**
	 * Replaces a specified sequence of characters with another sequence of characters in all genomes
	 * of the alignment.
	 *  
	 * @param oldSequence The old sequence of characters which has to be replaced.
	 * @param newSequence The new sequence of characters which has to replace the old one.
	 */
	public void replaceSequenceAlignment(String oldSequence, String newSequence) {

		// Check if the provided sequence is valid and has correct length
		if (isValidSequence(newSequence) && oldSequence.length() == newSequence.length()){
			// If valid, then change it everywhere
			for (String id: this.getIdentifiers()) {
				this.replaceSequenceGenome(id, oldSequence, newSequence);
			}
		} else {
			System.out.println("replaceSequenceAlignment was called with an invalid sequence. No change.");
			return;
		}
	}

	/* Getters and setters */

	/**
	 * Get the size of the alignment, i.e., the number of (ID, genome)-pairs present.
	 * 
	 * @return Integer denoting the number of pairs stored in the alignment.
	 */
	public int getSize() {
		return this.alignment.size();
	}

	/**
	 * Get the length of the genome sequences, i.e., the number of nucleotide characters. Note that all sequences are 
	 * assumed to have the same length within an alignment.
	 * 
	 * @return Integer denoting the length of the genome sequences of this alignment.
	 */
	public int getLengthGenomes() {
		return this.getGenomes().get(0).length();
	}

	/**
	 * Checks whether the given ID is present in the alignment or not.
	 * 
	 * @param identifier The ID we are going to look up in the alignment.
	 * @return Boolean specifying whether the given ID is present in the alignment or not.
	 */
	public boolean containsIdentifier(String identifier) {
		return this.getIdentifiers().contains(identifier);
	}

	/**
	 * Checks whether the given genome sequence is present in the alignment or not.
	 * 
	 * @param genome The genome sequence we are going to look up in the alignment.
	 * @return Boolean specifying whether the given genome is present in the alignment or not.
	 */
	public boolean containsGenome(String genome) {
		return this.getGenomes().contains(genome);
	}

	/**
	 * Method which gets the genome of the alignment given its identifier.
	 * 
	 * @param identifier The ID for which we want to look up the genome sequence.
	 * @return The genome sequence corresponding to the given ID.
	 */
	public String getGenome(String identifier) {
		// Check if the ID is present in the alignment
		if (this.containsIdentifier(identifier)){
			return this.alignment.get(identifier);
		} else {
			System.out.println("Sequence not found.");
			// By default, we return an empty string
			return "";
		}
	}

	/**
	 * Method which gets the ID of the alignment given its genome.
	 * 
	 * @param identifier The genome sequence for which we want to look up the ID.
	 * @return The ID corresponding to the given genome.
	 */
	public String getIdentifier(String genome) {
		// Get the identifier of the given sequence in the alignment
		if (this.containsGenome(genome)){
			for(Entry<String, String> entry: this.alignment.entrySet()) {
				// Return the key if it matches the desired sequence
				if(entry.getValue().equals(genome)) {
					return entry.getKey();
				}
			}
		} else {
			System.out.println("Identifier not found.");
		}
		// By default, we return an empty string
		return "";
	}

	/**
	 * Method which returns all the IDs (i.e., keys of the alignment HashMap) of this alignment.
	 * 
	 * @return An ArrayList holding all the IDs of the alignment.
	 */
	public ArrayList<String> getIdentifiers() {
		// Get the keys, as keyset, and convert to arraylist
		ArrayList<String> listOfKeys = new ArrayList<String>(this.alignment.keySet());
		return listOfKeys;
	}

	/**
	 * Method which returns all the genomes (i.e., values of the alignment HashMap) of this alignment.
	 * 
	 * @return An ArrayList holding all the genome sequences of the alignment.
	 */
	public ArrayList<String> getGenomes() {
		ArrayList<String> listOfSequences = new ArrayList<>(this.alignment.values());
		return listOfSequences;
	}

	/**
	 * Returns a deep copy of the HashMap storing the alignment as hashmap which can safely be edited and used in other methods.
	 * 
	 * @return A copy of the HashMap of this alignment.
	 */
	public HashMap<String, String> copyAlignment(){
		// 

		HashMap<String, String> copy = new HashMap<String, String>();
		for (Map.Entry<String, String> entry : this.alignment.entrySet()) {
			copy.put(entry.getKey(), entry.getValue());
		}
		return copy;
	}

	/**
	 * Creates a copy of this StandardAlignment object.
	 * 
	 * @return A new StandardAlignment object initialized with the currently stored alignment.
	 */
	public StandardAlignment copy(){
		StandardAlignment copy = new StandardAlignment(this.copyAlignment());
		return copy;
	}

	/**
	 * Setter for the alignment field.
	 * 
	 * @param alignment HashMap of (ID, sequences)-pairs we wish to store in the alignment field.
	 */
	public void setAlignment(HashMap<String, String> alignment){
		this.alignment = alignment;
	}

	/**
	 * Completely clears the contents of the alignment.
	 */
	public void clearAlignment() {
		this.alignment.clear();
	}

	/* Methods to compute the difference score */

	/**
	 * Computes the difference score (i.e., number of different nucleotides/positions) 
	 * between two pairs of genome sequences.
	 * 
	 * @param firstGenome A genome sequence
	 * @param secondGenome A genome sequence
	 * @return The difference score between the two given genome sequences.
	 */
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

	/**
	 * Computes the difference score of the alignment by adding up the difference score between all the genome
	 * sequences present in the alignment and a given, specified reference genome.
	 * 
	 * @param referenceGenome Genome sequence against which we are going to compare all genomes of the alignment.
	 * @return The difference score of this alignment.
	 */
	public int getDifferenceScore(String referenceGenome) {
		// Computes the difference score of this alignment.

		int score = 0;
		// Iterate over all the genomes in this list
		for (String genome : this.getGenomes()) {
			score += computeDifferenceScorePair(referenceGenome, genome);
		}
		return score;	
	}

	/**
	 * Computes the difference score of the alignment by adding up the difference score between all the genome
	 * sequences present in the alignment. The reference genome used is the first genome of the alignment
	 * @return The difference score of this alignment.
	 */
	public int getDifferenceScore() {
		// Get the reference genome, which is just the first one in the alignment
		ArrayList<String> allGenomes = this.getGenomes();
		String referenceGenome = allGenomes.get(0);
		return getDifferenceScore(referenceGenome);
	}

	/**
	 * Static method which checks if a given string is a valid genome sequence, meaning that
	 * it consists entirely of the nucleotide characters and a dot (used in SNiP view)
	 * 
	 * @param sequence A string, possibly genome sequence, which we are going to check with the method.
	 * @return Boolean indicating whether or not the given string represents part of a genome sequence.
	 */
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

	/**
	 * Method which checks if a given string is a valid genome to be added to the alignment. That means that it is a valid
	 * sequence of nucleotide characters and has the same length as all the other genome sequences present in the alignment.
	 * 
	 * @param genome A string which is possible a new genome sequence.
	 * @return Boolean indicating whether or not the given string represents valid genome.
	 */
	public boolean isValidGenome(String genome) {
		// First, check the length (assume all genomes in alignment are valid, i.e. have same length)
		int genomeLength = this.getLengthGenomes();
		if (genome.length() != genomeLength) {
			return false;
		}
		// Then, check if it's a correct sequence
		return isValidSequence(genome);
	}

	/**
	 * Method to display the first few (ID, sequence)-pairs of an alignment, useful for visualization 
	 * purposes in the Main method of the application. By default, the first three entries of the alignment are
	 * shown, and of each sequence, we show the first seventy nucleotide characters.
	 */
	public void display() {
		// Store the default amount of pairs + nucleotide characters to show here
		int defaultNumberOfLines       = 3;
		int defaultNumberOfNucleotides = 70;

		// Do nothing if there is nothing stored
		if (this.getSize() == 0) {
			System.out.println("This alignment is empty.");
			return;
		}

		// Get number of lines and nucleotide characters to show. Make sure we avoid IndexOutOfBounds issues
		int numberOfLines       = Math.min(this.getSize(), defaultNumberOfLines);
		int numberOfNucleotides = Math.min(this.getLengthGenomes(), defaultNumberOfNucleotides);

		// Display alignment entries:
		System.out.println("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
		for (int i = 0; i<numberOfLines; i++) {
			// Print the identifier
			String id = this.getIdentifiers().get(i);
			System.out.println(id);
			
			// Print  the genome sequence, add "(cont.)" if we do not reach the end of the sequence
			if (numberOfNucleotides == this.getLengthGenomes()) {
				System.out.println(this.getGenome(id).substring(0, numberOfNucleotides));
			} else {
				System.out.println(this.getGenome(id).substring(0, numberOfNucleotides) + " (cont.)");
			}
		}

		// In case this was not yet the end of the alignment, show to the screen that there are more entries
		if (!(numberOfLines == this.getSize())) {
			int remainingNumberOfLines = this.getSize() - numberOfLines;
			System.out.println("(continued by " + remainingNumberOfLines + " more entries)");
		}
		System.out.println("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
	}
}
