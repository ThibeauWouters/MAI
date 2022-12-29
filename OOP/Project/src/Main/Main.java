package Main;


import Alignments.*;
import Team.*;

public class Main {

	public static void main(String[] args) {
		/*
		 * Testing the code and assemblying everything together here.
		 */
		
		// TODO - specify path in the appropriate manner!
		
		System.out.println("--- Loading files. Creating repository and team.");
		
		// Read the fasta file
		FastaContents fasta = new FastaContents("hiv.fasta");
		
		// Create standard alignment from it
		StandardAlignment firstSA = new StandardAlignment(fasta);
		
		// Create an initial repo for this
		Repository repo = new Repository(firstSA);
		
		// Read the teams file
		Team team = new Team("team.txt", firstSA);
		
		// After the team has been set-up, update the team repository
		for (Employee employee : team.getTeam()) {
			if (employee instanceof AlignmentEditor){
				// View employee as editor:
				AlignmentEditor editor = (AlignmentEditor) employee; 
				repo.addTeamAlignment(editor, editor.getAlignment());
			}
		}
		
		/*
		 * Testing editing functions of Bioinformatician
		 */
		
		// TODO
		
		System.out.println("--- Editing alignments (TO DO)");
		
		/*
		 * Testing writing functionalities etc
		 */
		
		System.out.println("--- Writing data and reports.");

		// After the team has been set-up, update the team repository
		for (Employee employee : team.getTeam()) {
			// Bioinformatician's writers:
			if (employee instanceof AlignmentEditor){
				// View employee as editor:
				AlignmentEditor editor = (AlignmentEditor) employee; 
				editor.writeData(repo);
				editor.writeReport(repo);
			}
			
			// Teamleader's writers:
			if (employee instanceof TeamLead){
				// View employee as editor:
				TeamLead leader = (TeamLead) employee; 
				leader.writeData(repo);
				leader.writeReport(repo);
			}
		}

	}

}
