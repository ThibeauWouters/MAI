package team;

import datastructures.*;
import team.editors.Editor;

/**
 * This class implements the functionalities of the team leaders.
 * 
 * @author ThibeauWouters
 *
 */

public class TeamLead extends Employee {

	/* Constructor */
	
	/**
	 * Saves all instance variables of employee by calling the super constructor
	 */
	public TeamLead(String firstName, String lastName, int yearsOfExperience) {
		super("TeamLead", firstName, lastName, yearsOfExperience);
	}
	
	/* Methods */
	
	/**
	 * Promotes the personal alignment of an editor to become the optimal
	 * alignment of a repository.
	 * 
	 * @param repo The repository in which we are going to replace the optimal alignment.
	 * @param editor The editor of which we are going to promote the alignment to the optimal one.
	 */
	public void promoteAlignment(Repository repo, Editor editor) {
		repo.promoteAlignment(this, editor);
	}
	
	/**
	 * "Push" an alignment of an editor to the repository. That is, overwrite
	 * the alignment of that editor stored in the repository with the one
	 * currently stored as instance variable of the specified editor.
	 * 
	 * @param editor Editor object of which we are going to push to the repository.
	 * @param repo The repository to which we are going to push. 
	 */
	public void pushTeamAlignment(Editor editor, Repository repo) {
		repo.pushTeamAlignment(this, editor);
	}
	
	/**
	 * Overwrite the personal alignment of an editor with the optimal one
	 * stored in a repository.
	 *
	 * @param editor The editor whose alignment we are going to overwrite.
	 * @param repo The repository of which we are going to use the optimal alignment. 
	 */
	public void overwriteAlignment(Editor editor, Repository repo) {
		repo.overwriteAlignment(this, editor);
	}

	/**
	 * Write down all the alignments of the editors to the "Reports" directory of a Repository object
	 *
	 * @param repo The Repository object to which we will write the result
	 */
	public void writeData(Repository repo) {
		repo.writeData(this);
	}

	/**
	 * Write down all the scores of the alignments of the editors to the "Reports" directory of a Repository object
	 *
	 * @param repo The Repository object to which we will write the result
	 */
	public void writeReport(Repository repo) {
		repo.writeReport(this);
	}
}