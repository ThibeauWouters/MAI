package team;

import datastructures.*;
import team.editors.Editor;

public class TeamLead extends Employee {

	/* Variables */
	
	/* Constructor */
	
	public TeamLead(String firstName, String lastName, int yearsOfExperience) {
		super("TeamLead", firstName, lastName, yearsOfExperience);
	}
	
	/* Methods */
	
	/* Methods to push new optimal alignments to the repo, and check for improvements */
	
	public static boolean isValidScore(int score) {
		return score > 0;
	}
	
	public void promoteAlignment(Repository repo, Editor editor) {
		System.out.println("Promoting alignment of " + editor.getName() + " to optimal one.");
		repo.setOptimalSA(editor.copyAlignment());
	}
	
	public void checkForUpdates(Repository repo, Team team) {
		for (Employee employee : team.getTeam()) {
			if (employee instanceof Editor) {
				Editor editor = (Editor) employee;
				int score = editor.getScore();
				// Promote the alignment of this editor if it improves upon current optimal score
				if (score < repo.getOptimalScore() && isValidScore(score)) {
					System.out.println(editor.getName() + " has better score: " + editor.getScore() + " compared to current optimal: " + repo.getOptimalScore());
					promoteAlignment(repo, editor);
				}
			}
		}
	}
	
	/* Methods to edit other team member's alignments: */
	
	public void pushTeamAlignment(Editor editor, Repository repo) {
		repo.pushTeamAlignment(this, editor);
	}
	
	public void overwriteAlignment(Editor editor, Repository repo) {
		// Overwrites the alignment of the specified editor with the repo's optimal one
		repo.overwriteAlignment(editor);
	}
	
	/* Methods to write alignments and scores of all editors to files */
	
	public void writeData(Repository repo) {
		repo.writeData(this);
	}
	
	public void writeReport(Repository repo) {
		repo.writeReport(this);
	}
}
