package Alignments;

import java.util.*;

/**
 * An abstract class dictating what alignments should have implemented.
 * @author thibe
 *
 */
public interface AlignmentInterface {
	
	public String getGenome(String identifier);
	/*
	 * Simple function that returns the genome when given the identifier
	 */
	
	public String getIdentifier(String sequence);
	/*
	 * Simple function that returns the genome when given the identifier
	 */

	public ArrayList<String> searchGenomes(String subString);
	/* 
	 * A method to search through the genomes for a specific sequence of characters (substring). Returns
	 * the corresponding names/identifiers for those genomes in which the sequence can be found 
	 */
	
	public void replaceGenome(String oldId, String newId, String newSequence);
	/*
	 * Replace a genome in the alignment with a new sequence (i.e., a genome that was
	 * previously not part of the alignment and that has a name / identifier that was not part of
	 * the initial FASTA file)
	 */
	
	public void replaceSequenceGenome(String identifier, String oldSequence, String newSequence);
	/*
	 * in a given genome, replace all occurrences of a given sequence of characters by a new
	 * sequence of characters (without changing the total length of the genome)
	 */
	
	public void replaceSequenceAlignment(String oldSequence, String newSequence);
	/*
	 * Same as above, but do this for all genomes in the alignment
	 */
	
	public void addGenome(String identifier, String sequence);
	/*
	 * Add a genome with its corresponding name / identifier
	 */

	public void removeGenome(String identifier);
	/*
	 * Remove a genome, based on its name / identifier
	 */
	
	/* Other functionalities: */
	
	public int getLengthAlignment();
	/*
	 * Get the length (number of entries in the alignment)
	 */
	
	public ArrayList<String> getIdentifiers();
	/*
	 * Get all the identifiers in an ArrayList
	 */
	
	public ArrayList<String> getSequences();
	/*
	 * Get all the sequences in an ArrayList
	 */
}
