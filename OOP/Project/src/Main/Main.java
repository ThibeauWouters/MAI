package main;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.InputMismatchException;
import java.util.Properties;
import java.util.Scanner;

import alignments.*;
import datastructures.Repository;
import team.*;
import team.editors.Bioinformatician;
import team.editors.Editor;

/**
 * Main program showing how the different classes fit together with each other.

 * @author ThibeauWouters
 *
 */

public class Main {

	public static void main(String[] args) {

		/*
		 * Prepare everything to read and write files with proper relative paths for any OS
		 */

		// Initialize names of files to be read
		String teamFileName  = "";
		String fastaFileName = "";

		// Locate the path of this project with File and Path classes
		File file = new File("");
		String currentPath = file.getAbsolutePath();
		Path myPath = Paths.get(currentPath);
		System.out.println("Current path is: " + myPath.toString());

		// Check whether the reports and backup folder already exist, and if not, create them
		File reportsDirectory = new File(currentPath, "Reports");
		File backupsDirectory = new File(currentPath, "Backups");
		File[] dirs = new File[] {reportsDirectory, backupsDirectory};
		for (File dir : dirs) {
			if (!dir.exists()) {
				dir.mkdir();
			} 
		}

		// Get the filenames from the properties file:
		Properties prop = new Properties();
		File propertiesFilePath = new File(currentPath, "config.properties");

		try (InputStream input = new FileInputStream(propertiesFilePath)) {
			prop.load(input);

			// Get the filenames
			teamFileName  = prop.getProperty("teamfilename");
			fastaFileName = prop.getProperty("fastafilename");

			System.out.println(teamFileName);
			System.out.println(fastaFileName);

		} catch (IOException ioe) {
			System.out.println("IOException when reading properties file :");
			ioe.printStackTrace();
		} catch (Exception e) {
			System.out.println("Something went wrong when reading properties file:");
			e.printStackTrace();
		}

		/*
		 * Read the .fasta and team files
		 */

		System.out.println("--- Loading files. Creating repository and team.");

		// Read the fasta file
		System.out.println("Reading alignment from " + fastaFileName);
		FastaContents fasta = new FastaContents(fastaFileName);

		// Create standard alignment and SNiP alignment from it
		StandardAlignment firstSA   = new StandardAlignment(fasta);
		StandardAlignment firstSNiP = new SNiPAlignment(fasta, ">1990.B.TH.90.BK132.AY173951");

		// Create an initial repo for this
		Repository repo = new Repository(firstSA, reportsDirectory, backupsDirectory);

		/*
		 * Read the teams files
		 */
		System.out.println("Reading team from " + teamFileName);
		ArrayList<Employee> teamArrayList = new ArrayList<Employee>();

		// Check if fileName corresponds to a TXT file (final 4 characters are ".txt")
		String extension = teamFileName.substring(teamFileName.length()-4, teamFileName.length());
		if (!(extension.equals(".txt"))) {
			System.out.println("Team file must be .txt file. Exiting program.");
			System.exit(0);
		}
		// Scan all lines of this file
		try(Scanner input = new Scanner(new FileReader(teamFileName));){ 
			while(input.hasNext()) { 
				try {
					// Read in the next lines and define a new employee with it
					Employee newEmployee;

					// Read the contents of this employee
					String employeeType               = input.next();
					String firstName                  = input.next();
					String lastName                   = input.next();
					String yearsOfExperienceString    = input.next();

					// Try to parse the final field as an integer, otherwise use "0" as default number of years of experience
					int yearsOfExperience = 0;
					try {
						yearsOfExperience = Integer.parseInt(yearsOfExperienceString);
					} catch (NumberFormatException e) {
						System.out.println("Could not interpret yearsOfExperience as a number. Defaulting to 0.");
					}
					// Call the appropriate constructor, depending on the employee type:
					// (In future updates, one can add additional employee types here . . .)
					switch (employeeType) {
					case "TeamLead":         newEmployee = new TeamLead(firstName, lastName, yearsOfExperience); break;
					case "Bioinformatician": newEmployee = new Bioinformatician(firstName, lastName, yearsOfExperience, firstSA.copy()); break;
					case "TechnicalSupport": newEmployee = new TechnicalSupport(firstName, lastName, yearsOfExperience); break; 
					default: System.out.println("Employeetype not recognized. Ignoring this line"); continue;
					}

					// Add this new employee to the team
					teamArrayList.add(newEmployee);

				} catch (InputMismatchException mm) {
					// In case the input is invalid:
					System.out.println("Invalid entry in teams file. Exiting program."); 
					mm.printStackTrace();
					System.exit(0);
					break; 
				} 
			}  
		} catch (FileNotFoundException e) {
			// In case the rights to file are invalid:
			System.out.println("Error: teams file not found. Exiting program."); 
			e.printStackTrace();
			System.exit(0); 
		} catch (Exception e) {
			// Catch any other exception
			System.out.println("Unexpected error occurred while reading teams file. Exiting program.");
			e.printStackTrace();
			System.exit(0); 
		}

		Team team = new Team(teamArrayList);

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
			repo.addTeamAlignment(editor);
		}

		// Show the initial optimal alignment:
		System.out.println("The starting optimal standard alignment has header:");
		repo.displayOptimalSA();
		System.out.println("There are " + firstSA.getSize() + " genomes in the fasta file.");
		System.out.println("All genomes have a length of " + firstSA.getLengthGenomes());
		System.out.println("The corresponding SNiP alignment has referenceID: " + repo.getReferenceID());
		System.out.println("The corresponding SNiP alignment looks like:");
		repo.displayOptimalSNiP();
		System.out.println("The current optimal score is: " + repo.getOptimalScore());
		System.out.println("The difference score can be computed in SNiP view as well: " + firstSNiP.getDifferenceScore());
		String first  = "A..T.C";
		String second = "A..GG.";
		int testScore = SNiPAlignment.computeDifferenceScorePair(first, second);
		System.out.println("For example, difference between " + first + " and " + second + " is " + testScore);

		/*
		 * Testing editing functions of Bioinformatician
		 */

		System.out.println("--- Bioinformaticians enter the lab and start working...");
		System.out.println(bio.getName() + " is going to edit. Before, their alignment looks as follows:");
		bio.displayAlignment();
		// Show method to replace sequences
		System.out.println(bio.getName() +  " edits all TTT's to AAA's in their alignment. Result:");
		bio.replaceSequenceAlignment("TTT", "AAA");
		bio.displayAlignment();
		// Show method that searches for particular substrings
		String subString = "AAAAAAAAAA";
		System.out.println(bio.getName() + " now gathers all IDs of genomes containing " + subString + ". Result:");
		ArrayList<String> test = bio.searchGenomes(subString);
		System.out.println(test);
		System.out.println("Some edits are not allowed: replacing TTT with AA (different length)");
		// A message is printed by the following call
		bio.replaceSequenceAlignment("TTT", "AA");
		System.out.println("Another example: replacing TTT with GXC (invalid character)");
		// A message is printed by the following call
		bio.replaceSequenceAlignment("TTT", "GXC");
		String deleteID = ">1992.C.BR.92.BR025";
		System.out.println(bio.getName() + " is going to delete genome with ID: " + deleteID + ". Result:");
		// For illustration purposes, get a genome from the above .fasta file
		String newSequence = firstSA.getGenome(deleteID);
		bio.removeGenome(deleteID);
		bio.displayAlignment();
		String replaceID = ">1994.A2.CY.94.94CY017";
		System.out.println(bio.getName() + " is going to replace genome with ID: " + replaceID);
		bio.replaceGenome(replaceID, newSequence);
		bio.displayAlignment();
		String newID = "new";
		System.out.println(bio.getName() + " adds that same genome again with ID: " + newID + ". Result:");
		bio.addGenome(newID, newSequence);
		bio.displayAlignment();
		System.out.println("Not allowed: adding genomes with IDs already used in alignment:");
		// The following prints an error message, as newID is already used as an ID:
		bio.addGenome(newID, newSequence);
		System.out.println(bio.getName() +  " now has difference score: " + bio.getScore());

		/*
		 * Testing writing functionalities for bioinformaticians
		 */

		System.out.println("--- All bioinformaticians record their progress so far and write reports.");

		// After the team has been set-up, update the team repository
		for (Editor editor : team.getEditors()) { 
			editor.writeData(reportsDirectory);
			editor.writeReport(reportsDirectory);
		}

		/*
		 * Testing team lead's functionalities
		 */

		System.out.println("--- Team leaders enter the lab and check new alignments ...");
		System.out.println(leader.getName() + " is going to write data and reports.");
		leader.writeData(repo);
		leader.writeReport(repo);

		System.out.println(leader.getName() + " sees a better alignment...");
		leader.promoteAlignment(repo, bio);

		System.out.println("--- After the promotion, we check the current optimal alignment again...");
		System.out.println("Now, the optimal standard alignment has header:");
		repo.displayOptimalSA();
		System.out.println("The corresponding SNiP alignment has referenceID: " + repo.getReferenceID());
		System.out.println("The corresponding SNiP alignment has header:");
		repo.displayOptimalSNiP();


		System.out.println("--- What does alignments of the team members in the repository look like?");
		HashMap<Editor, StandardAlignment> teamAlignments = repo.getTeamAlignments();
		for (Editor editor : teamAlignments.keySet()) {
			System.out.println(editor.getName() + " has this alignment in repository:");
			teamAlignments.get(editor).display();
		}

		System.out.println("--- A team leader pushes everyone's work to the repository...");
		for (Editor editor: team.getEditors()) {
			leader.pushTeamAlignment(editor, repo);
		}


		System.out.println("--- How does the repository look like afterwards?");
		teamAlignments = repo.getTeamAlignments();
		for (Editor editor : teamAlignments.keySet()) {
			System.out.println(editor.getName() + " has this alignment in repository:");
			teamAlignments.get(editor).display();
		}

		System.out.println("--- A team leader decides to overwrite another bioinformatician's alignment...");
		Editor secondEditor = team.getEditors().get(1);
		leader.overwriteAlignment(secondEditor, repo);


		System.out.println("--- How does the repository look like afterwards?");
		teamAlignments = repo.getTeamAlignments();
		for (Editor editor : teamAlignments.keySet()) {
			System.out.println(editor.getName() + " has this alignment in the repository:");
			teamAlignments.get(editor).display();
		}

		/*
		 * Show the functionalities of the technical staff:
		 */

		System.out.println("--- The technical staff is called...");
		tech.createBackup(repo);
		tech.clear(repo);
		System.out.println("--- What does the repository currently look like?");
		System.out.println("The optimal aligment:");
		repo.displayOptimalSA();
		System.out.println("The SNiP aligment:");
		repo.displayOptimalSNiP();
		for (Editor editor : teamAlignments.keySet()) {
			System.out.println(editor.getName() + " has this alignment in the repository:");
			teamAlignments.get(editor).display();
		}

		System.out.println("--- The backup is loaded again");
		tech.load(repo);
		System.out.println("--- What does the repository currently look like?");
		System.out.println("The optimal aligment:");
		repo.displayOptimalSA();
		System.out.println("The SNiP aligment:");
		repo.displayOptimalSNiP();
		for (Editor editor : teamAlignments.keySet()) {
			System.out.println(editor.getName() + " has this alignment in the repository:");
			teamAlignments.get(editor).display();
		}
		System.out.println("--- When was the backup made?");
		tech.displayBackupTimestamp();
	}
}
