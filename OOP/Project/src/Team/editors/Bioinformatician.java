package team.editors;

import alignments.*;

public class Bioinformatician extends Editor {
	
	/* Variables */
	
	// Additional variables could appear here . . .
	
	/* Constructor */
	
	public Bioinformatician(String firstName, String lastName, int yearsOfExperience, StandardAlignment sa) {
		super("Bioinformatician", firstName, lastName, yearsOfExperience, sa);
	}
	
	/* Methods to edit the alignments */
	// Note: we give the bioinformaticians the right to all edit functionalities
	
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
