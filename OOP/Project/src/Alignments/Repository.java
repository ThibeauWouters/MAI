package Alignments;

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

import Team.*;

public class Repository {
	
	/* Variables */
	private int optimalScore;
	private StandardAlignment optimalSA;
	// TODO - change with relative path!
	private String backupFolderName = "src/Backups/";
	private SNiPAlignment optimalSNiP;
	private HashMap<AlignmentEditor, StandardAlignment> teamAlignments = new HashMap<AlignmentEditor, StandardAlignment>();
	
	/* Constructor */
	
	public Repository(StandardAlignment sa) {
		// The repository is initialized with a standard alignment.
		// When this alignment is set, it automatically stores the SNiP version as well as its score.
		this.setOptimalSA(sa);
	}

	/* Methods */
	
	public void displayOptimalSNiP() {
		for (String id: this.getOptimalSNiP().getIdentifiers()) {
			System.out.println(id);
			String genome = this.getOptimalSNiP().getGenome(id);
			System.out.println(genome);
		}
	}
	
	/* Team alignments functionalities: */
	
	public void addTeamAlignment(AlignmentEditor editor, StandardAlignment sa) {
		this.teamAlignments.put(editor, sa);
	}
	
	public void removeTeamAlignment(Employee employee) {
		// TODO - check if present
		this.teamAlignments.remove(employee);
	}
	
	/* Repo & Alignments processing methods */
	
	public void overwriteAlignment(AlignmentEditor editor, StandardAlignment sa) {
		// Overwrites the alignment of the given editor with the given alignment.
		
		editor.setAlignment(sa);
		// Also updates the alignment in the repo
		this.teamAlignments.replace(editor, sa);
	}
	
	public void overwriteAlignment(AlignmentEditor editor) {
		// Overwrites the alignment of the given editor with the optimal one.
		
		this.teamAlignments.replace(editor, this.getOptimalSA());
	}
	
	
	public void clear(TechnicalSupport tech) {
		// Completely clears the repository
		
		// Show who's going to clear the repo
		System.out.println(tech.getName() + " is clearing the repository.");
		
		// Delete optimal alignment
		this.setOptimalSA(null);
		// Delete alignment of each editor
		for (AlignmentEditor editor : this.getTeamAlignments().keySet()) {
			this.overwriteAlignment(editor, null);
		}
	}
	
	public void backup(TechnicalSupport tech, String fileName) {
		// Creates a backup of this repository
		
		// Show who's going to create a backup
		System.out.println(tech.getName() + " is making backup of the repository at " + tech.getBackupTimestamp() + ".");
		
		// Get the correct location of the file
		fileName = this.getBackupFolderName() + fileName;
		System.out.println("Saving backup to " + fileName);
		
		try(BufferedWriter bw = new BufferedWriter(new FileWriter(fileName))){			
			// Save the optimal alignment first
			bw.write("Optimal" + "\n");
			copyAlignmentToFile(bw, this.getOptimalSA());
			// Iterate over the editors of the team
			for (Entry<AlignmentEditor, StandardAlignment> entry : this.getTeamAlignments().entrySet()) {
				// Get the current entry's editor and his/her alignment
			    AlignmentEditor editor        = entry.getKey();
			    StandardAlignment alignment   = entry.getValue();
			    // Write editor's name to the file
			    bw.write(editor.getName() + "\n");
			    // Then, write his/her alignment to file
			    copyAlignmentToFile(bw, alignment);
			}
		}
		catch (FileNotFoundException fnf) {
			System.out.println("File not found: " + fnf);
		} 
		catch (IOException ioe) {
			System.out.println("IOexception: " + ioe);
		} 
	}
	
	public void load(TechnicalSupport tech, String fileName) {
		// Load the repo from a given file, and save 
		// TODO - what if there is a team member for which there is no information in the backup-file?
		
		System.out.println(tech.getName() +  " is going to load backup data from " + fileName);
		
		// Get editor's names:
		ArrayList<String> editorNames = new ArrayList<String>();
		for (AlignmentEditor editor : this.getTeamAlignments().keySet()) {
			editorNames.add(editor.getName());
		}
		
		// Scan all lines, add the identifier and genome sequence to respective ArrayLists.
		try(Scanner input = new Scanner(new FileReader(fileName));){ 
			while(input.hasNext()) { 
				try {
					// Get name or optimal, and create empty hashmap to store the next alignment
					String alignmentName = input.next();
					HashMap<String, String> hmap = new HashMap<String, String>();
					
					// Read in the first (id, seq) pair (alignment cannot be empty)
					// TODO - what if empty alignment???
					String id  = input.next();
					String seq = input.next();
					
					// Reading next line which can either be genome ID or name of next editor:
					String line = input.next();
					while (!editorNames.contains(line)) {
						// If line was not a name, it was an ID
						id = line;
						// And the next line will be a seq
						seq = input.next();
						// Store that pair
						hmap.put(id, seq);
						// Again, continue, but check if name or if (id, seq) pair (while-loop)
						line = input.next();
					}
					// If we reach here, the above while loop ended, and the alignment of an editor is finished
					// First, creat a SA object from the HashMap we built:
					StandardAlignment alignment = new StandardAlignment(hmap); 
					// Find the correct editor (or optimal), then give its alignment
					if (alignmentName.contains("Optimal")) {
						// Reset the optimal alignment
						this.setOptimalSA(alignment);
					} else {
						// TODO - can it be that the alignment name is no longer in team?
						// If it was an editor's alignment, find the correct object and store
						for (AlignmentEditor editor : this.getTeamAlignments().keySet()) {
							if (editor.getName().equals(alignmentName)) {
								// Overwrite the editor's repo (this changes both repo as well as object's alignment)
								this.overwriteAlignment(editor, alignment);
							}							
						}
					}
				} catch (InputMismatchException mismatch) {
					// In case the input is invalid:
					System.out.println("Invalid entry in file found, stop reading input."); 
					break; 
				} 
			}  
		} catch (FileNotFoundException e) {
			// In case the rights to file are invalid:
			// TODO: is this indeed the error in case we don't have the right to access the file?
			System.out.println("Error: file not found or you don't have the rights to access this file?"); 
			System.out.println("Exiting program");
			System.exit(0); 
		} catch (Exception e) {
			// Catch any other exception
			System.out.println("Unexpected error occurred: " + e);
			System.out.println("Exiting program");
			System.exit(0); 
		}
	}
	
	/* Methods to write to files */
	
	// Auxiliary function (private, only visible here): copy an entire alignment to a file
	private static void copyAlignmentToFile(BufferedWriter bw, StandardAlignment sa) throws IOException {
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
			for (Entry<AlignmentEditor, StandardAlignment> entry : this.getTeamAlignments().entrySet()) {
				// Get the current entry's editor and his/her alignment
			    AlignmentEditor editor        = entry.getKey();
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
	public void writeData(AlignmentEditor editor) {
		String folderName = "src/";
		String fileName = folderName + editor.getFirstName() + "_" + editor.getLastName() + ".alignment.txt";
		System.out.println("Saving data of " + editor.getName() + " to " + fileName);
		
		try(BufferedWriter bw = new BufferedWriter(new FileWriter(fileName))){
			copyAlignmentToFile(bw, editor.getAlignment());
		}
		catch (FileNotFoundException fnf) {
			System.out.println("File not found when writing alignment: " + fnf);
		} 
		catch (IOException ioe) {
			System.out.println("IOexception when writing alignment: " + ioe);
		} 
	}
	
	
	// Editor writes his/her alignment to file
	public void writeReport(AlignmentEditor editor) {
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
			for (AlignmentEditor editor : this.getTeamAlignments().keySet()) {
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

	public HashMap<AlignmentEditor, StandardAlignment> getTeamAlignments() {
		return teamAlignments;
	}
	
	public void replaceTeamAlignments(AlignmentEditor editor, StandardAlignment sa) {
		this.teamAlignments.replace(editor, sa);
	}
	
	public void updateTeamAlignments() {
		// Iterate over all editors and its alignments, and resets their alignment
		
		// TODO - what if the member was not present yet?
		for (AlignmentEditor editor : this.getTeamAlignments().keySet()) {
			StandardAlignment sa = editor.getAlignment();
			this.replaceTeamAlignments(editor, sa);
		}
	}

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

	public void setOptimalSNiP(SNiPAlignment optimalSNiP) {
		// Reset the current optimal SNiP alignment based on current optimal SA
		this.optimalSNiP = optimalSNiP;
		// Update optimal score after updating optimal alignment
		this.updateOptimalScore();
	}

	public void setOptimalSA(StandardAlignment optimalSA) {
//		// TODO - update this
//		// Updates the optimal SA. Automatically updates the SNiP and optimal score
//		this.optimalSA = optimalSA;
//		// Create a SNiP alignment from this SA, and update instance variable

//		SNiPAlignment snip = new SNiPAlignment(optimalSA);
//		this.setOptimalSNiP(snip);
//		this.updateOptimalScore();
	}
	
	public String getBackupFolderName() {
		return backupFolderName;
	}
	
	public void setBackupFolderName(String backupFolderName) {
		this.backupFolderName = backupFolderName;
	}
	
	// Getters for the optimal alingments (SA and SNiP) is not visible outside this class!
	private StandardAlignment getOptimalSA() {
		return optimalSA;
	}
	
	private SNiPAlignment getOptimalSNiP() {
		return optimalSNiP;
	}
}
