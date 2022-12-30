package main;


import java.util.HashMap;

import alignments.*;
import datastructures.Repository;
import team.*;
import team.editors.Bioinformatician;
import team.editors.Editor;

public class Main {

	public static void main(String[] args) {

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
		
		// For future convenience, save a leader, a bioinformatician, and a technical support for demonstration
		// (we require that every team has at least one of each in this demonstration)
		Bioinformatician bio  = team.getBioinformaticians().get(0);
		TeamLead leader       = team.getTeamLeads().get(0);
		TechnicalSupport tech = team.getTechnicalSupports().get(0);
		
		// Display the team members:
		System.out.println("Our team:");
		team.displayTeamMembers();

		// After the team has been set-up, update the team repository
		for (Editor editor : team.getEditors()) {
			repo.addTeamAlignment(editor, editor.copyAlignment());
		}
		
		// Show the initial optimal alignment:
		System.out.println("The starting optimal standard alignment has header:");
		repo.displayOptimalSA();
		System.out.println("The corresponding SNiP alignment has referenceID: " + repo.getReferenceID());
		System.out.println("The corresponding SNiP alignment has header:");
		repo.displayOptimalSNiP();
		System.out.println("The current optimal score is: " + repo.getOptimalScore());

		/*
		 * Testing editing functions of Bioinformatician
		 */

		System.out.println("--- Bioinformaticians enter the lab and start working...");
				
		// After the team has been set-up, update the team repository
		String someID = bio.copyAlignment().getIdentifiers().get(0);
		System.out.println(bio.getName() + " is going to edit. Before, genome " + someID + " is:");
		System.out.println(bio.copyAlignment().getGenome(someID));
		System.out.println(bio.getName() +  " edits all T's to A's in his alignment. Result:");
		bio.replaceSequenceAlignment("T", "A");
		System.out.println(bio.copyAlignment().getGenome(someID));
		System.out.println(bio.getName() +  " now has difference score: " + bio.getScore());

		/*
		 * Testing writing functionalities etc
		 */

		System.out.println("--- All bioinformaticians record their progress so far.");

		// After the team has been set-up, update the team repository
		for (Editor editor : team.getEditors()) { 
			editor.writeData(repo);
			editor.writeReport(repo);
		}
		
		/*
		 * Testing team lead's functionalities
		 */

		System.out.println("--- Team leaders enter the lab and check all new alignments for improvements...");
		System.out.println(leader.getName() + " is going to write data and reports.");
		leader.writeData(repo);
		leader.writeReport(repo);
		
		System.out.println(leader.getName() + " is going to check for improvement in difference score.");
		leader.checkForUpdates(repo, team);
		
		System.out.println("--- After the work, we check the current optimal alignment again...");
		System.out.println("Now, the optimal standard alignment has header:");
		repo.displayOptimalSA();
		System.out.println("The corresponding SNiP alignment has referenceID: " + repo.getReferenceID());
		System.out.println("The corresponding SNiP alignment has header:");
		repo.displayOptimalSNiP();

		
		System.out.println("--- What does alignments of the team members in the repository look like?");
		HashMap<Editor, StandardAlignment> teamAlignments = repo.getTeamAlignments();
		for (Editor editor : teamAlignments.keySet()) {
			System.out.println(editor.getName() + " has this alignment in repo:");
			teamAlignments.get(editor).display();
		}
		
		System.out.println("--- A team leader decides to push everyone's work to the repo...");
		for (Editor editor: team.getEditors()) {
			leader.pushTeamAlignment(editor, repo);
		}
		
		
		System.out.println("--- How does the repository look like afterwards?");
		teamAlignments = repo.getTeamAlignments();
		for (Editor editor : teamAlignments.keySet()) {
			System.out.println(editor.getName() + " has this alignment in repo:");
			teamAlignments.get(editor).display();
		}
		
		System.out.println("--- A team leader decides to overwrite another bioinformatician's alignment...");
		// TODO - allowed to assume more than 1 editor???
		Editor secondEditor = team.getEditors().get(1);
		leader.overwriteAlignment(secondEditor, repo);
		
		
		System.out.println("--- How does the repository look like afterwards?");
		teamAlignments = repo.getTeamAlignments();
		for (Editor editor : teamAlignments.keySet()) {
			System.out.println(editor.getName() + " has this alignment in repo:");
			teamAlignments.get(editor).display();
		}
		
		
		System.out.println("--- Technical staff is called (TO DO)");
		
		// TODO
		
//		// After the team has been set-up, update the team repository
//		for (Employee employee : team.getTeam()) {
//			// Get first tech support employee
//			if (employee instanceof TechnicalSupport){
//				TechnicalSupport tech = (TechnicalSupport) employee; 
//			}
//		}
		
	}

}
