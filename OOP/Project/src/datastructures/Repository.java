package datastructures;

import java.io.BufferedWriter;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.InputMismatchException;
import java.util.Scanner;
import java.util.Map.Entry;

import alignments.*;
import team.*;
import team.editors.Editor;

public class Repository {

	/* Variables */
	private int optimalScore;
	private StandardAlignment optimalSA;
	private SNiPAlignment optimalSNiP;
	private String referenceGenome;
	private String referenceID;
	private HashMap<Editor, StandardAlignment> teamAlignments = new HashMap<Editor, StandardAlignment>();

	/* Constructor */

	public Repository(StandardAlignment sa) {
		// The repository is initialized with a standard alignment.
		// When this alignment is set, it automatically stores the SNiP version as well as its score.
		this.setOptimalSA(sa);
	}

	/* Methods */

	/* Team alignments functionalities: */

	/* Add, remove or replace alignments in the team repo */

	public void addTeamAlignment(Editor editor, StandardAlignment sa) {
		this.teamAlignments.put(editor, sa.copy());
	}

	public void removeTeamAlignment(Employee employee) {
		// TODO - check if present
		this.teamAlignments.remove(employee);
	}

	public void replaceTeamAlignment(Editor editor, StandardAlignment sa) {
		this.removeTeamAlignment(editor);
		this.addTeamAlignment(editor, sa.copy());
	}

	/* Repo & Alignments processing methods */

	public void pushTeamAlignment(TeamLead leader, Editor editor) {
		System.out.println(leader.getName() + " pushes alignment of " + editor.getName() + " to repository.");
		this.replaceTeamAlignment(editor, editor.copyAlignment());
	}
	
	public void overwriteAlignment(Editor editor, StandardAlignment sa) {
		// Overwrites the alignment of the given editor with the given alignment.

		editor.setAlignment(sa);
		// Also updates the alignment in the repo
		this.replaceTeamAlignment(editor, sa);
	}

	public void overwriteAlignment(Editor editor) {
		// Overwrites the alignment of the given editor with the optimal one.

		this.overwriteAlignment(editor, this.getOptimalSA());
	}


	public void clear(TechnicalSupport tech) {
		// Completely clears the repository

		// Show who's going to clear the repo
		System.out.println(tech.getName() + " is clearing the repository.");

		// Delete optimal alignment
		this.setOptimalSA(null);
		// Delete alignment of each editor
		for (Editor editor : this.getTeamAlignments().keySet()) {
			this.overwriteAlignment(editor, null);
		}
	}

	/* Write backup files */

	// Backup the alignment of a single editor:
	public void backup(TechnicalSupport tech, Editor editor) {
		// Creates a backup of this repository

		// Get the filename, which uses TechSupport and Editor's names
		String fileName = "src/" + tech.getBackupString(editor);

		// Show who's going to create a backup
		System.out.println(tech.getName() + " is making backup of the repository at " + tech.getBackupTimestamp() + ".");
		System.out.println("Saving backup to " + fileName);

		// Get the Editor's alignment:
		StandardAlignment alignment = this.getTeamAlignments().get(editor);

		// Copy this alignment to the specified fileName;
		copyAlignmentToFile(fileName, alignment);
	}

	// Do the above for all the editors:
	public void backupEditors(TechnicalSupport tech) {
		// Iterate over the editors of the team
		for (Editor editor : this.getTeamAlignments().keySet()) {
			this.backup(tech, editor);
		}
	}

	// Create a backup of the optimal
	public void backupOptimal(TechnicalSupport tech) {
		
		// Backup of standard version:
		String folderName = "src/";
		String fileName = folderName + tech.getBackupString("OptimalSA");
		// Save alignment to the file:
		copyAlignmentToFile(fileName, this.getOptimalSA().copy());
		
		// Backup of SNiP version:
		fileName = folderName + tech.getBackupString("OptimalSNiP");
		copyAlignmentToFile(fileName, this.getOptimalSNiP().copy());
	}
	
	public void backup(TechnicalSupport tech) {
		// Create a backup of everything: optimal alignment (and SNiP)
		
		this.backupOptimal(tech);
		this.backupEditors(tech);
	}
	
	/* Load from backup files */

	public void loadOptimalSA(TechnicalSupport tech) {
		// Load alignment from the backup for optimal alignment
		
		// Read the specified backup file, and create StandardAlignment object from it	
		String folderName = "src/";
		String fileName = folderName + tech.getBackupString("OptimalSA");
		
		FastaContents fasta = new FastaContents(fileName);
		StandardAlignment sa = new StandardAlignment(fasta);

		// Save the new alignment
		this.setOptimalSA(sa);
	}


	public void load(TechnicalSupport tech, Editor editor) {
		// Load alignment from the backup for a single person
		
		// Read the specified backup file, and create StandardAlignment object from it	
		String folderName = "src/";
		String fileName = folderName + tech.getBackupString(editor);
		
		FastaContents fasta = new FastaContents(fileName);
		StandardAlignment sa = new StandardAlignment(fasta);

		// Save the new alignment into editor's copy and repository copy
		this.replaceTeamAlignment(editor, sa);
		editor.setAlignment(sa);
	}
	
	public void loadEditors(TechnicalSupport tech) {
		// Iterate over the editors of the team
		for (Editor editor : this.getTeamAlignments().keySet()) {
			this.load(tech, editor);
		}
	}
	
	public void load(TechnicalSupport tech, String fileName) {
		// Load the optimal alignment from backup and also editor's backups
		
		this.loadOptimalSA(tech);
		this.loadEditors(tech);
	}

	/* Methods to write to files */

	// Auxiliary function (private, only visible here): copy an entire alignment to a file, given the fileName
	private static void copyAlignmentToFile(String fileName, StandardAlignment sa) {
		// Save alignment to the file:
		try(BufferedWriter bw = new BufferedWriter(new FileWriter(fileName))){			
			// Write down the alignment
			for (String identifier : sa.getIdentifiers()) {
				String genome = sa.getGenome(identifier);
				bw.write(identifier + "\n");
				bw.write(genome + "\n");
			}
		}
		catch (FileNotFoundException fnf) {
			System.out.println("File not found: " + fnf);
		} 
		catch (IOException ioe) {
			System.out.println("IOexception: " + ioe);
		}	
	} 
	
	// Auxiliary function: same as above, but we are given the bufferedwriter
	private static void copyAlignmentToFile(BufferedWriter bw, StandardAlignment sa) throws IOException {
		// Write down the alignment
		for (String identifier : sa.getIdentifiers()) {
			String genome = sa.getGenome(identifier);
			bw.write(identifier + "\n");
			bw.write(genome + "\n");
		}
	} 

	// Teamleader writes data to file
	public void writeData(TeamLead leader) {
		String folderName = "src/";
		String fileName = folderName + leader.getFirstName() + "_" + leader.getLastName() + ".alignment.txt";
		System.out.println("Saving data of team to " + fileName);

		try(BufferedWriter bw = new BufferedWriter(new FileWriter(fileName))){
			// Write down the optimal standard alignment
			bw.write("Optimal" + "\n");
			copyAlignmentToFile(bw, this.getOptimalSA());
			// Iterate over the editors of the team
			for (Entry<Editor, StandardAlignment> entry : this.getTeamAlignments().entrySet()) {
				// Get the current entry's editor and his/her alignment
				Editor editor        = entry.getKey();
				StandardAlignment alignment   = entry.getValue();
				// Write editor's name to the file
				bw.write(editor.getName() + "\n");
				// Then, write his/her alignment to file
				copyAlignmentToFile(bw, alignment);
			}
		}
		catch (FileNotFoundException fnf) {
			System.out.println("File not found when writing alignment: " + fnf);
		} 
		catch (IOException ioe) {
			System.out.println("IOexception when writing alignment: " + ioe);
		} 
	}

	// Editor writes his/her alignment to file
	public void writeData(Editor editor) {
		String folderName = "src/";
		String fileName = folderName + editor.getFirstName() + "_" + editor.getLastName() + ".alignment.txt";
		System.out.println("Saving data of " + editor.getName() + " to " + fileName);

		try(BufferedWriter bw = new BufferedWriter(new FileWriter(fileName))){
			copyAlignmentToFile(bw, editor.copyAlignment());
		}
		catch (FileNotFoundException fnf) {
			System.out.println("File not found when writing alignment: " + fnf);
		} 
		catch (IOException ioe) {
			System.out.println("IOexception when writing alignment: " + ioe);
		} 
	}


	// Editor writes his/her alignment to file
	public void writeReport(Editor editor) {
		String folderName = "src/";
		String fileName = folderName + editor.getFirstName() + "_" + editor.getLastName() + ".score.txt";
		System.out.println("Saving score of " + editor.getName() + " to " + fileName);

		try(BufferedWriter bw = new BufferedWriter(new FileWriter(fileName))){
			bw.write(editor.getName() + "\n");
			bw.write(editor.getScore() + "\n");
		}
		catch (FileNotFoundException fnf) {
			System.out.println("File not found when writing alignment: " + fnf);
		} 
		catch (IOException ioe) {
			System.out.println("IOexception when writing alignment: " + ioe);
		} 
	}


	// Teamleader writes scores to file
	public void writeReport(TeamLead leader) {
		String folderName = "src/";
		String fileName = folderName + leader.getFirstName() + "_" + leader.getLastName() + ".score.txt";
		System.out.println("Saving report of team to " + fileName);

		try(BufferedWriter bw = new BufferedWriter(new FileWriter(fileName))){
			bw.write("Optimal" + "\n");
			bw.write(this.getOptimalScore() + "\n");
			// Iterate over the editors of the team
			for (Editor editor : this.getTeamAlignments().keySet()) {
				int score = editor.getScore();
				bw.write(editor.getName() + "\n");
				bw.write(score + "\n");
			}
		}
		catch (FileNotFoundException fnf) {
			System.out.println("File not found when writing alignment: " + fnf);
		} 
		catch (IOException ioe) {
			System.out.println("IOexception when writing alignment: " + ioe);
		} 
	}

	/* Getters and setters */

	public HashMap<Editor, StandardAlignment> getTeamAlignments() {
		return this.teamAlignments;
	}

	public void updateTeamAlignments() {
		// Iterate over all editors and its alignments, and resets their alignment

		// TODO - what if the member was not present yet?
		for (Editor editor : this.getTeamAlignments().keySet()) {
			StandardAlignment sa = editor.copyAlignment();
			this.replaceTeamAlignment(editor, sa);
		}
	}

	/* Getters and setters */ 

	public int getOptimalScore() {
		return optimalScore;
	}

	public void setOptimalScore(int optimalScore) {
		this.optimalScore = optimalScore;
	}

	public void updateOptimalScore() {
		// Set the difference score by recomputing it from the current optimal SA

		int score = this.optimalSA.getDifferenceScore();
		this.setOptimalScore(score);
	}

	private StandardAlignment getOptimalSA() {
		// Getters for the optimal alingments (SA and SNiP) is not visible outside this class!
		
		return optimalSA;
	}

	public void setOptimalSA(StandardAlignment optimalSA) {
		// Updates the optimal standard alignment

		// Updates the optimal SA. Automatically updates the SNiP and optimal score
		this.optimalSA = optimalSA.copy();

		// Create a SNiP alignment from this SA, and update instance variable
		SNiPAlignment snip = new SNiPAlignment(optimalSA);
		this.setOptimalSNiP(snip);
		this.updateOptimalScore();
	}

	private SNiPAlignment getOptimalSNiP() {
		return optimalSNiP.copy();
	}

	public void setOptimalSNiP(SNiPAlignment optimalSNiP) {
		// Reset the current optimal SNiP alignment based on current optimal SA
		this.optimalSNiP = optimalSNiP.copy();
		// Also update the referenceGenome and referenceID
		this.referenceGenome = this.optimalSNiP.getReferenceGenome();
		this.referenceID     = this.optimalSNiP.getReferenceID();
		// Update optimal score after updating optimal alignment
		this.updateOptimalScore();
	}
	
	/* Visualization (extras) */
	
	public void displayOptimalSA() {
		// Display the header of the optimal alignment (standard alignment)
		this.optimalSA.display();
	}
	
	public void displayOptimalSNiP() {
		// Display the header of the optimal alignment (standard alignment)
		this.optimalSNiP.display();
	}

	public String getReferenceGenome() {
		return referenceGenome;
	}

	public void setReferenceGenome(String referenceGenome) {
		this.referenceGenome = referenceGenome;
	}

	public String getReferenceID() {
		return referenceID;
	}

	public void setReferenceID(String referenceID) {
		this.referenceID = referenceID;
	}
}
