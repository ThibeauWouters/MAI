package Team;

import Alignments.*;

public class Bioinformatician extends Employee {

	
	/* Variables */
	StandardAlignment alignment; 
	
	/* Constructor */
	
	public Bioinformatician(String firstName, String lastName, int yearsOfExperience, StandardAlignment sa) {
		super("Bioinformatician", firstName, lastName, yearsOfExperience);
		this.alignment = sa;
	}
	
	/* Methods */
	
}
