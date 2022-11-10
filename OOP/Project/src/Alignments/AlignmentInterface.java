package Alignments;

import java.util.*;

/**
 * An abstract class dictating what alignments should have implemented.
 * @author thibe
 *
 */
public interface AlignmentInterface {

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
	
//	public void replaceSequenceAlignment();
//	/*
//	 * Same as above, but do this for all genomes in the alignment
//	 */
//	
//	public void addGenome();
//	/*
//	 * Add a genome with its corresponding name / identifier
//	 */
//	
//	public void removeGenome();
//	/*
//	 * Remove a genome, based on its name / identifier
//	 */
//	
//	// ... other functions? 
//	// Ideas: give all identifiers, check if identifier is present, give the length of the alignment,...
//	// get sequence (given identifier), get identifier given sequence.

}
