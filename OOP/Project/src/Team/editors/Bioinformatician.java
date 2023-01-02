package team.editors;

import java.util.ArrayList;

import alignments.*;

/**
 * This is a class implementing the functionalities of the Bioinformaticians.
 * It is a subclass of the abstract Editor class. Currently, all possible ways to edit
 * an alignment are implemented here. Future extensions of the program can easily allow
 * users to create different subclasses of Editor, for different employee types which
 * also will have different ways to edit alignments. Methods are easily added or
 * removed as they simply call an edit function on the instance variable of the
 * editor holding their personal copy of an alignment.
 * 
 * @author ThibeauWouters
 *
 */
public class Bioinformatician extends Editor {
	
	/* Constructor */
	
	/**
	 * The constructor works similarly as the Editor constructor. 
	 */
	public Bioinformatician(String firstName, String lastName, int yearsOfExperience, StandardAlignment sa) {
		super("Bioinformatician", firstName, lastName, yearsOfExperience, sa);
	}
	
	/* Methods to edit the alignments */
	
	/**
	 * For all methods implemented here, please look at the
	 * StandardAlignment class for their implementation. 
	 */
	
	public ArrayList<String> searchGenomes(String subString){
		return this.alignment.searchGenomes(subString);
	}
	
	public void addGenome(String identifier, String sequence) {
		this.alignment.addGenome(identifier, sequence);
	}

	public void removeGenome(String identifier) {
		this.alignment.removeGenome(identifier);
	}
	
	public void replaceGenome(String oldId, String newId, String newSequence) {
		this.alignment.replaceGenome(oldId, newId, newSequence);
	}
	
	public void replaceGenome(String id, String newSequence) {
		this.replaceGenome(id, id, newSequence);
	}
	
	public void replaceSequenceGenome(String identifier, String oldSequence, String newSequence) {
		this.alignment.replaceSequenceGenome(identifier, oldSequence, newSequence);
	}
	
	public void replaceSequenceAlignment(String oldSequence, String newSequence) {
		this.alignment.replaceSequenceAlignment(oldSequence, newSequence);
	}
}
