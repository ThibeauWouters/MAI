package Team;

import Alignments.*;

public class TeamLead extends Employee {

	/* Variables */
	
	/* Constructor */
	
	public TeamLead(String firstName, String lastName, int yearsOfExperience) {
		super("TeamLead", firstName, lastName, yearsOfExperience);
	}
	
	/* Methods */
	
	/* Methods to push new optimal alignments to the repo, and check for improvements */
	
	public void promoteAlignment(Repository repo, AlignmentEditor editor) {
		System.out.println("Promoting alignment of " + editor.getName() + " to optimal one.");
		repo.setOptimalSA(editor.getAlignment());
	}
	
	public static boolean isValidScore(int score) {
		return score > 0;
	}
	
	public void checkForUpdates(Repository repo, Team team) {
		for (Employee employee : team.getTeam()) {
			if (employee instanceof AlignmentEditor) {
				AlignmentEditor editor = (AlignmentEditor) employee;
				int score = editor.getScore();
				// Promote the alignment of this editor if it improves upon current optimal score
				if (score < repo.getOptimalScore() && isValidScore(score)) {
					promoteAlignment(repo, editor);
				}
			}
		}
	}
	
	/* Methods to write alignments and scores of all editors to files */
	
	public void writeData(Repository repo) {
		repo.writeData(this);
	}
	
	public void writeReport(Repository repo) {
		repo.writeReport(this);
	}
	
	public void overwriteAlignment(AlignmentEditor editor, Repository repo) {
		// Overwrites the alignment of the specified editor with the repo's optimal one
		repo.overwriteAlignment(editor);
	}
}
