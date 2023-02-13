package alignments;

import java.util.*;
import java.util.Map.Entry;

/**
 * This is a class that implements all functionalities related to the standard alignment.
 * An instance of this class stores the alignment as a Java Hashmap, with the identifiers stored as keys, and the
 * corresponding values being their genome sequences. Operations such as editing sequences and adding/removing
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
	 * in a genome sequence, namely the four nucleotides and a dot which is used by SNiPAlignments, which is
	 * a subclass of this class.
	 */
	public final static Set<Character> NUCLEOTIDES = new HashSet<>(Arrays.asList('A', 'C', 'T', 'G', '.')); 


	/* Constructor */

	/**
	 * Constructor which takes the contents of a FASTA file, after it has been read by the AlignmentsIO class and saved
	 * as an instance of that class. It copies the contenst of the alignment in that object to the alignment
	 * as defined in this class.
	 * 
	 * @param fasta AlignmentsIO object holding the alignment read from a file.
	 */
	public StandardAlignment(AlignmentsIO fasta) {
		// Load the contents of the fasta file into the alignment, make sure not called with a null object
		if (!(fasta == null)) {
			// Iterate over all the ID and genome pairs, and store them in the hashmap as (key, value)-pairs.
			for(int i=0; i<fasta.getIdentifiers().size(); i++) {
				this.alignment.put(fasta.getIdentifiers().get(i), fasta.getGenomes().get(i));
			}
		}
	}

	/**
	 * Constructor taking a HashMap, holding an alignment as (key, value)-pairs, and copying
	 * its to the alignment field of this object.
	 * 
	 * @param hmap HashMap of identifier and sequence pairs.
	 */
	public StandardAlignment(HashMap<String, String> hmap) {
		// Load the contents of the fasta file into the alignment:
		if (!(hmap == null)) {
			// Iterate over all keys, i.e. IDs of an alignment
			for (String id : hmap.keySet()) {
				// Get their value
				String genome = hmap.get(id);
				// Store in the alignment of this object
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
	 * after editing it such that it again represents genome sequences instead of differences between sequences (that
	 * means, we put nucleotide characters at the positions of the dots again). See the SNiPAlignment class for
	 * the implementation of the methods used here.
	 * 
	 * @param snip A SNiPAlignment object
	 */
	public StandardAlignment(SNiPAlignment snip) {
		// Copy the alignment
		if (!(snip == null)) {
			this.alignment = snip.copyAlignment();
			// Then edit such that we have genome rather than difference for all IDs
			for (String id : snip.getIdentifiers()) {
				// Get the value in SNiP version: this is the difference view
				String difference = snip.getGenome(id);
				// Recreate the standard view from the difference view
				String genome     = snip.getOriginalGenome(difference);
				// Store the (ID, genome)-pair in this alignment
				this.alignment.replace(id, genome);
			}
		}
	}

	/**
	 * Empty constructor, used for constructors of the subclass SNiPAlignment.
	 */
	public StandardAlignment() {}

	/* Methods */

	/**
	 * Searches through all the genomes of the alignment and returns those genome sequences
	 * which contain a given substring as part of their sequence.
	 * 
	 * @param subString Sequence with which we will look for matches within all genomes of the alignment
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
	 * Adds a new (identifier, genome)-pair into the alignment. Checks whether the given genome has a valid
	 * representation before storing it (see below).
	 * 
	 * @param identifier The ID of the genome to be added.
	 * @param genome The genome sequence to be added.
	 */
	public void addGenome(String identifier, String genome) {
		// Check if the given genome sequence is valid
		if (!(this.isValidGenome(genome))) {
			System.out.println("addGenome is called with an invalid sequence. No change.");
		// Check if the ID is already used by this alignment - not allowed to have duplicates!
		} else if (this.containsIdentifier(identifier)) {
			System.out.println("ID already present in alignment. No change.");
		} else {
			// If OK, store the new (ID, genome)-pair
			this.alignment.put(identifier, genome);
		}
	}

	/**
	 * Deletes the (ID, genome)-pair with specified identifier from this alignment.
	 * 
	 * @param identifier The ID of the genome which we wish to delete.
	 */
	public void removeGenome(String identifier) {
		// Check if the ID is in this alignment
		if (this.containsIdentifier(identifier)) {
			this.alignment.remove(identifier);
		} else {
			// The ID is not here - print an error message
			System.out.println("removeGenome called with ID not present in alignment.");
		}
	}

	/**
	 * Replaces an existing (ID, sequence)-pair with a new pair.
	 * 
	 * @param oldId The ID of the existing genome sequence we wish to replace
	 * @param newId The ID of the new genome sequence we wish to add
	 * @param newGenome The new genome sequence to be added to the alignment
	 */
	public void replaceGenome(String oldId, String newId, String newGenome) {
		// Check if the given genome is valid
		if (!this.isValidGenome(newGenome)) {
			System.out.println("replaceGenome is called with an invalid sequence.");
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
	 * Replaces an existing (ID, sequence)-pair in the alignment, but reuses the old ID.
	 * 
	 * @param oldId The ID of the genome sequence we wish to replace 
	 * @param newSequence The new genome sequence to be added.
	 */
	public void replaceGenome(String oldId, String newSequence) {
		this.alignment.replace(oldId, newSequence);
	}

	/**
	 * Replaces a particular sequence of nucleotides of a given identifier specifying a genome
	 * with another sequence of nucleotide characters.
	 * 
	 * @param identifier The ID of the genome in which we want to replace characters.
	 * @param oldSequence The old sequence of characters which has to be replaced.
	 * @param newSequence The new sequence of characters which has to replace the old sequence of characters.
	 */
	public void replaceSequenceGenome(String identifier, String oldSequence, String newSequence) {
		// First, check if the provided sequence is valid, and if the replacement won't change the length of genomes
		if (isValidSequence(newSequence) && oldSequence.length() == newSequence.length()){
			// Get the original genome
			String oldGenome = this.getGenome(identifier);
			// Do the replacement of the sequences
			String newGenome = oldGenome.replaceAll(oldSequence, newSequence);
			// Store in alignment again
			this.replaceGenome(identifier, newGenome);
		} else {
			System.out.println("replaceSequenceGenome was called with an invalid sequence. No change.");
		}
	}

	/**
	 * Replaces a specified sequence of characters with another sequence of characters in all genomes
	 * of this alignment.
	 *  
	 * @param oldSequence The old sequence of characters which has to be replaced.
	 * @param newSequence The new sequence of characters which has to replace the old sequence of characters.
	 */
	public void replaceSequenceAlignment(String oldSequence, String newSequence) {
		// Check if the provided sequence is valid and if the replacement keeps the same length of genomes
		if (isValidSequence(newSequence) && oldSequence.length() == newSequence.length()){
			// If valid, then change it by calling the above method on all the genomes
			for (String id: this.getIdentifiers()) {
				this.replaceSequenceGenome(id, oldSequence, newSequence);
			}
		} else {
			System.out.println("replaceSequenceAlignment was called with an invalid sequence. No change.");
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
	 * Method which returns the genome of the alignment given its identifier.
	 * 
	 * @param identifier The ID of the genome sequence we wish to find.
	 * @return The genome sequence corresponding to the given ID.
	 */
	public String getGenome(String identifier) {
		// Check if the ID is present in the alignment
		if (this.containsIdentifier(identifier)){
			// If present, return it
			return this.alignment.get(identifier);
		} else {
			// By default, we return the empty string
			System.out.println("Sequence not found.");
			return "";
		}
	}

	/**
	 * Method which gets the ID of the alignment given its genome.
	 * 
	 * @param genome The genome sequence for which we want to look up the ID.
	 * @return The ID corresponding to the given genome.
	 */
	public String getIdentifier(String genome) {
		// Check if genome is in the alignment
		if (this.containsGenome(genome)){
			// Find the key that corresponds to this genome as value
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
		// Get the keys, using the keyset method, and convert to ArrayList
		return new ArrayList<String>(this.alignment.keySet());
	}

	/**
	 * Method which returns all the genomes (i.e., values of the alignment HashMap) of this alignment.
	 * 
	 * @return An ArrayList holding all the genome sequences of the alignment.
	 */
	public ArrayList<String> getGenomes() {
		return new ArrayList<>(this.alignment.values());
	}

	/**
	 * Returns a deep copy of the HashMap storing the alignment as hashmap. Using this method,
	 * we can safely get the alignment here to use it in other methods.
	 * 
	 * @return A copy of the HashMap of this alignment.
	 */
	public HashMap<String, String> copyAlignment(){
		// Create an empty HashMap
		HashMap<String, String> copy = new HashMap<String, String>();
		// Iterate over all (key, value)-pairs of the current alignment, and store in the "copy" HashMap
		for (Map.Entry<String, String> entry : this.alignment.entrySet()) {
			copy.put(entry.getKey(), entry.getValue());
		}
		return copy;
	}

	/**
	 * Creates a StandardAlignment object as copy of this StandardAlignment object.
	 * 
	 * @return A new StandardAlignment object initialized with the currently stored alignment.
	 */
	public StandardAlignment copy(){
		return new StandardAlignment(this.copyAlignment());
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
	 * Computes the difference score (i.e., number of different nucleotides/positions) between
	 * two pairs of genome sequences.
	 * 
	 * @param firstGenome A first genome sequence
	 * @param secondGenome A second genome sequence
	 * @return The difference score between the two given genome sequences.
	 */
	public static int computeDifferenceScorePair(String firstGenome, String secondGenome) {
		// Initialize the score as zero
		int score = 0;

		// Iterate over all characters of both genomes. If they are different, we add 1
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
		// Initialize score as zero
		int score = 0;
		// Iterate over all the genomes in this list, compute difference score with the provided reference genome
		for (String genome : this.getGenomes()) {
			score += computeDifferenceScorePair(referenceGenome, genome);
		}
		return score;	
	}

	/**
	 * Computes the difference score of the alignment by adding up the difference score between all the genome
	 * sequences present in the alignment. The reference genome used is by default the first genome of the alignment
	 * @return The difference score of this alignment.
	 */
	public int getDifferenceScore() {
		// Get the reference genome, which is just the first one in the alignment
		ArrayList<String> allGenomes = this.getGenomes();
		String referenceGenome = allGenomes.get(0);
		// Then compute the difference score using the method above
		return getDifferenceScore(referenceGenome);
	}

	/**
	 * Static method which checks if a given string is a valid genome sequence, meaning that
	 * it consists entirely of the nucleotide characters and a dot (used in SNiP view).
	 * 
	 * @param sequence A string, possibly genome sequence, which we are going to check with the method.
	 * @return Boolean indicating whether the given string represents part of a genome sequence.
	 */
	public static boolean isValidSequence(String sequence) {
		// Get all characters as an array
		char[] allChars = sequence.toCharArray();
		// Convert them to a set
		Set<Character> setOfChars = new HashSet<>();
		for (char c : allChars) {
			setOfChars.add(c);
		}
		// The sequence is valid of its characters is a subset of the allowed characters
		return NUCLEOTIDES.containsAll(setOfChars);
	}

	/**
	 * Method which checks if a given string is a valid genome to be added to the alignment. That means that it is a valid
	 * sequence of nucleotide characters and has the same length as all the other genome sequences present in the alignment.
	 * 
	 * @param genome A string which is possible a new genome sequence.
	 * @return Boolean indicating whether the given string represents valid genome.
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
		// Get number of lines and nucleotide characters to show. Make sure we avoid IndexOutOfBounds
		int numberOfLines       = Math.min(this.getSize(), defaultNumberOfLines);
		int numberOfNucleotides = Math.min(this.getLengthGenomes(), defaultNumberOfNucleotides);

		// Display alignment entries:
		System.out.println("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
		try {
			for (int i = 0; i<numberOfLines; i++) {
				// Print the identifier
				String id = this.getIdentifiers().get(i);
				System.out.println(id);

				// Print the genome sequence, add "(cont.)" if we do not reach the end of the sequence
				if (numberOfNucleotides == this.getLengthGenomes()) {
					System.out.println(this.getGenome(id).substring(0, numberOfNucleotides));
				} else {
					System.out.println(this.getGenome(id).substring(0, numberOfNucleotides) + " (cont.)");
				}
			}
		} catch (IndexOutOfBoundsException e){
			System.out.println("Index out of bounds when trying to display alignment.");
			e.printStackTrace();
		}
		// In case this was not yet the end of the alignment, show to the screen that there are more entries
		if (!(numberOfLines == this.getSize())) {
			int remainingNumberOfLines = this.getSize() - numberOfLines;
			System.out.println("(continued by " + remainingNumberOfLines + " more entries)");
		}
		System.out.println("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
	}
}
