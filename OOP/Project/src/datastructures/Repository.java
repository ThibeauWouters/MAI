package datastructures;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map.Entry;

import alignments.*;
import team.*;
import team.editors.*;

/**
 * This class implements all functionalities and aspects related to the repository
 * of the bioinformatics team. Moreover, it implements all functionalities of the team
 * members which deal with stored alignments in the repository.
 * 
 * @author ThibeauWouters
 */
public class Repository {

	/* Variables */
	/**
	 * reportsDirectory stores the path to the "Reports" directory as a File object.
	 * Using this, we can easily construct correct relative paths 
	 * when writing data and reports to files
	 */
	private File reportsDirectory;
	/**
	 * backupsDirectory stores the path to the "Backups" directory as a File object.
	 * Using this, we can easily construct correct relative paths 
	 * when writing backups to files
	 */
	private File backupsDirectory;
	/**
	 * optimalScore stores the current value of the difference score of the optimal alignment
	 */
	private int optimalScore;
	/**
	 * optimalSA stores the current standard view of the optimal alignment
	 */
	private StandardAlignment optimalSA;
	/**
	 * optimalSNiP stores the current SNiP view of the optimal alignment
	 */
	private SNiPAlignment optimalSNiP;
	/**
	 * referenceGenome stores the standard view of the reference genome used
	 * when constructing the SNiP view of the optimal alignment
	 */
	private String referenceGenome;
	/**
	 * referenceID stores the ID of the reference genome used
	 * when constructing the SNiP view of the optimal alignment
	 */
	private String referenceID;
	/**
	 * teamAlignments is a HashMap with the Editors working on private copies of alignments
	 * as keys and a remote copy stored in this repository as their corresponding value
	 */
	private HashMap<Editor, StandardAlignment> teamAlignments = new HashMap<Editor, StandardAlignment>();

	/* Constants */
	
	/**
	 * BACKUP_EXTENSION stores an extension always used for backup files
	 */
	private final static String BACKUP_EXTENSION = ".backup.txt";
	/**
	 * DATA_EXTENSION stores an extension always used for data files
	 */
	private final static String DATA_EXTENSION   = ".alignment.txt";
	/**
	 * REPORT_EXTENSION stores an extension always used for report files
	 */
	private final static String REPORT_EXTENSION = ".score.txt";

	/* Constructor */

	/**
	 * Constructor takes a StandardAlignment object, which is going to be the initial optimal
	 * alignment in the repository, and File objects storing the path to the "Backup" and 
	 * "Reports" directories where we will save files.
	 * 
	 * @param sa A StandardAlignment object serving as the initial optimal alignment
	 * @param reportsDirectory A File object storing the path to the "Reports" directory
	 * @param backupsDirectory A File object storing the path to the "Backups" directory
	 */
	public Repository(StandardAlignment sa, File reportsDirectory, File backupsDirectory) {
		// Store the given SA, create its SNiP view, and update the relevant fields
		this.optimalSA        = sa.copy();
		this.optimalSNiP      = new SNiPAlignment(optimalSA);
		this.referenceID      = this.optimalSNiP.getReferenceID();
		this.referenceGenome  = this.optimalSNiP.getReferenceGenome();
		this.optimalScore     = this.optimalSA.getDifferenceScore();		
		// Save the reports and backups directory locations
		this.reportsDirectory = reportsDirectory;
		this.backupsDirectory = backupsDirectory;
	}

	/* Methods */

	/* Add, remove or replace alignments in the team repo */

	/**
	 * Add a pair of editor and his/her personal alignment to the HashMap
	 * storing all the remote copies of the alignments.
	 * 
	 * @param editor An Editor object not yet present in the repository
	 */
	public void addTeamAlignment(Editor editor) {
		// Check if editor is already present in the HashMap
		if(this.getTeamAlignments().keySet().contains(editor)) {
			System.out.println("Editor already present in repository of team alignments. Use replaceTeamAlignment instead.");
			return;
		} else {
			// If not, add the editor as key and his/her alignment as value to the HashMap
			this.teamAlignments.put(editor, editor.copyAlignment());
		}
	}

	/**
	 * Remove an editor and his/her remote copy of alignment from the repository if present.
	 * 
	 * @param editor An Editor object which we wish to remove from the repository.
	 */
	public void removeTeamAlignment(Editor editor) {
		// Check if the given editor is present in HashMap
		if(!(this.getTeamAlignments().keySet().contains(editor))) {
			System.out.println("Editor already present in repository of team alignments.");
			return;
		} else {
			this.teamAlignments.remove(editor);
		}
	}

	/**
	 * Replaces (i.e., updates) the remote copy of an editor in the repository.
	 * 
	 * @param editor An Editor object for which we wish to update the copy in the repository.
	 * @param sa A StandardAlignment object that is going to overwrite the remote copy in the repository.
	 */
	public void replaceTeamAlignment(Editor editor, StandardAlignment sa) {
		this.teamAlignments.replace(editor, sa.copy());
	}

	/* Repo & Alignments processing methods */

	/**
	 * Method which can be called by a leader of the team who wishes to push the current 
	 * personal copy of an alignment of an editor to the repository.
	 * 
	 * @param leader A TeamLead object who calls this method, and pushes to the repository
	 * @param editor An Editor object of which the personal copy gets pushed to the repository
	 */
	public void pushTeamAlignment(TeamLead leader, Editor editor) {
		System.out.println(leader.getName() + " pushes alignment of " + editor.getName() + " to repository.");
		this.replaceTeamAlignment(editor, editor.copyAlignment());
	}
	
	public void promoteAlignment(TeamLead leader, Editor editor) {
		System.out.println(leader.getName() + " promotes alignment of " + editor.getName() + " to optimal one.");
		this.setOptimalSA(editor.copyAlignment());
	}

	/**
	 * Method which can be called by either a team leader or a technical staff member which
	 * overwrites the personal copy of an editor with the alignment that is currently stored
	 * as the optimal alignment in the repository.
	 * 
	 * @param employee Either a team leader or a technical staff member calling this method.
	 * @param editor The Editor object of which the personal alignment is going to be overwritten.
	 */
	public void overwriteAlignment(Employee employee, Editor editor) {
		// Check if employee has access rights to the repository
		if (!(employee.hasRepositoryAccess())) {
			System.out.println(employee.getName() + " has no access to repository.");
			return;
		}
		// Show who's going to overwrite who's alignment
		System.out.println(employee.getName() + " overwrites alignment of " + editor.getName());
		// Update the editor's personal alignment
		editor.setAlignment(this.getOptimalSA().copy());
		// Also updates the alignment in the repo
		this.replaceTeamAlignment(editor, this.getOptimalSA().copy());
	}

	/**
	 * Method called by technical staff member which clears the current contents of the repository
	 * 
	 * @param tech The technical staff member which is clearing the repository.
	 */
	public void clear(TechnicalSupport tech) {
		// Show who's going to clear the repo
		System.out.println(tech.getName() + " is clearing the repository.");
		// Clear both SA and SNiP views of the optimal alignment
		this.optimalSA.clearAlignment();
		this.optimalSNiP.clearAlignment();
		// Clear all remote copies of alignments of editors
		for (Editor editor : this.getTeamAlignments().keySet()) {
			this.teamAlignments.get(editor).clearAlignment();
		}
	}

	/* Write backup files */

	/**
	 * Create a backup of a single editor's alignment.
	 * 
	 * @param tech The technical staff member calling this method.
	 * @param editor The editor of which a backup is made of his/her alignment
	 */
	public void backup(TechnicalSupport tech, Editor editor) {
		// Show who's making the backup
		System.out.println(tech.getName() + " is creating a backup of alignment of " + editor.getName());
		// Get path to the "Backup" directory
		String currentPath = backupsDirectory.getAbsolutePath();
		// Create a filename in this folder for this specific editor
		File path = new File(currentPath, editor.getFileString(getBackupExtension()));
		// Get the correct relative path
		String fileName = path.getAbsolutePath();
		// Print statement for visualization
		System.out.println("Saving backup to " + fileName);
		// Get the Editor's alignment:
		StandardAlignment alignment = this.getTeamAlignments().get(editor);
		// Copy the entire alignment to the specified file
		copyAlignmentToFile(fileName, alignment);
	}

	/**
	 * Create a backup of all alignments of editors stored in the repository. 
	 * 
	 * @param tech The technical staff member creating the backup
	 */
	public void backupEditors(TechnicalSupport tech) {
		// Iterate over the editors of the team
		for (Editor editor : this.getTeamAlignments().keySet()) {
			// Call backup method for each of them
			this.backup(tech, editor);
		}
	}

	/**
	 * Create a backup of the optimal alignment.
	 * 
	 * @param tech The technical staff member who is creating the backup.
	 */
	public void backupOptimal(TechnicalSupport tech) {
		// Show who's making the backup
		System.out.println(tech.getName() + " is creating a backup of the optimal alignment.");
		// Get path to the "Backups" directory
		String currentPath = backupsDirectory.getAbsolutePath();
		// Get the path to the desired file
		File path = new File(currentPath, "OptimalSA.txt");
		// Get the filename as string
		String fileName = path.getAbsolutePath();
		// Show where we will save the backup to
		System.out.println("Saving backup to " + fileName);
		// Copy entire alignment to the desired file:
		copyAlignmentToFile(fileName, this.getOptimalSA().copy());
		// Repeat for the SNiP version:
		path = new File(currentPath, "OptimalSNiP.txt");
		fileName = path.getAbsolutePath();
		copyAlignmentToFile(fileName, this.getOptimalSNiP().copy());
	}

	/**
	 * Create a backup of the entire repository.
	 * 
	 * @param tech The technical staff member creating the backup.
	 */
	public void backup(TechnicalSupport tech) {
		this.backupOptimal(tech);
		this.backupEditors(tech);
	}

	/* Load from backup files */

	/**
	 * Load the StandardAlignment saved in the backup and overwrite the optimal
	 * alignment currently stored in the repository with it.
	 *  
	 * @param tech 
	 */
	public void loadOptimalSA(TechnicalSupport tech) {
		// Show who's loading the backup
		System.out.println(tech.getName() + " is loading a backup of the optimal alignment.");
		// Read the specified backup file (get correct path)	
		String currentPath = backupsDirectory.getAbsolutePath();
		File backupFile = new File(currentPath, "OptimalSA.txt");
		String fileName = backupFile.getAbsolutePath();
		// Check if a backup file of the optimal SA was already made before continuing
		if (!(backupFile.exists())) {
			System.out.println("There does not exist a backup file of OptimalSA. Exiting method.");
			return;
		}
		// Load in and create standard alignment from backup file
		FastaContents fasta  = new FastaContents(fileName);
		StandardAlignment sa = new StandardAlignment(fasta);
		// In case something unexpectedly went wrong, make sure not to overwrite with empty alignment
		if (sa.getSize() == 0) {
			System.out.println("Backup optimal SA is empty. Exiting method.");
			return;
		}
		// If nothing went wrong, overwrite the optimal alignment
		this.setOptimalSA(sa);
	}

	/**
	 * Load an editor's alignment from their backup file.
	 * 
	 * @param tech
	 * @param editor
	 */
	public void load(TechnicalSupport tech, Editor editor) {
		// Show who's loading the backup
		System.out.println(tech.getName() + " is loading a backup of the alignment of " + editor.getName());
		// Create the relative path pointing to the backup file	
		String currentPath = backupsDirectory.getAbsolutePath();
		File backupFile = new File(currentPath, editor.getFileString(getBackupExtension()));
		String fileName = backupFile.getAbsolutePath();
		if (!(backupFile.exists())) {
			System.out.println("There does not exist a backup file for " + editor.getName() + ". Exiting method.");
			return;
		}
		// Read the backup file, and create StandardAlignment object from it
		FastaContents fasta  = new FastaContents(fileName);
		StandardAlignment sa = new StandardAlignment(fasta);
		// In case something unexpectedly went wrong, make sure not to overwrite with empty alignment
		if (sa.getSize() == 0) {
			System.out.println("Backup optimal SA is empty. Exiting method.");
			return;
		}
		// Save the new alignment into editor's copy and repository copy
		this.replaceTeamAlignment(editor, sa);
		editor.setAlignment(sa);
	}

	/**
	 * Load in the backup alignments for all the editors again.
	 * 
	 * @param tech The technical staff member loading the backup.
	 */
	public void loadEditors(TechnicalSupport tech) {
		// Iterate over the editors of the team
		for (Editor editor : this.getTeamAlignments().keySet()) {
			// Load the backup for each of them
			this.load(tech, editor);
		}
	}

	/**
	 * Load a backup of the entire repository, i.e. optimal and personal alignments.
	 * 
	 * @param tech The technical staff member loading the backup files.
	 */
	public void load(TechnicalSupport tech) {
		this.loadOptimalSA(tech);
		this.loadEditors(tech);
	}

	/* Methods to write to files */

	/**
	 * Auxiliary method that writes down a StandardAlignment object to a specified file, 
	 * using the same format as the .fasta files (that is, printing each identifier and its
	 * corresponding genome sequence).
	 * 
	 * @param fileName The name of the file to which we will write the alignment
	 * @param sa The alignment we which to save to the file
	 */
	public static void copyAlignmentToFile(String fileName, StandardAlignment sa) {
		// Save alignment to the file:
		try(BufferedWriter bw = new BufferedWriter(new FileWriter(fileName))){			
			// Iterate over all IDs
			for (String identifier : sa.getIdentifiers()) {
				// Get the corresponding genome
				String genome = sa.getGenome(identifier);
				// Write the ID to the file
				bw.write(identifier + "\n");
				// Write the genome sequence to the file
				bw.write(genome + "\n");
			}
		}
		catch (FileNotFoundException fnf) {
			System.out.println("File not found error occurred while writing alignment.");
			fnf.printStackTrace();
		} 
		catch (IOException ioe) {
			System.out.println("IOexception occurred while writing alignment.");
			ioe.printStackTrace();
		}	
	} 

	/**
	 * Writes an alignment to a file with a currently open bufferedwriter.
	 * 
	 * @param bw Currently opene bufferedwriter which is saving several alignments.
	 * @param sa The current alignment we which to write to the bufferedwriter.
	 * @throws IOException During writing, an IOException can occur.
	 */
	public static void copyAlignmentToFile(BufferedWriter bw, StandardAlignment sa) throws IOException {
		// Write down the alignment
		for (String identifier : sa.getIdentifiers()) {
			String genome = sa.getGenome(identifier);
			bw.write(identifier + "\n");
			bw.write(genome + "\n");
		}
	} 

	/**
	 * Method which allows team leaders to write down all alignments of editors to a file.
	 * 
	 * @param leader Team leader who is writing down the data.
	 */
	public void writeData(TeamLead leader) {
		// Get the correct relative path to the file where we will save data	
		String currentPath = reportsDirectory.getAbsolutePath();
		File path = new File(currentPath, leader.getFileString(getDataExtension()));
		String fileName = path.getAbsolutePath();
		// Visualize the location to the screen:
		System.out.println("Saving data of team to " + fileName);

		// Open a new bufferedwriter to write down all data
		try(BufferedWriter bw = new BufferedWriter(new FileWriter(fileName))){
			// Write down the optimal standard alignment first
			bw.write("Optimal" + "\n");
			copyAlignmentToFile(bw, this.getOptimalSA());
			// Then, iterate over the editors of the team
			for (Entry<Editor, StandardAlignment> entry : this.getTeamAlignments().entrySet()) {
				// Get the current entry's editor and his/her alignment
				Editor editor = entry.getKey();
				StandardAlignment alignment   = entry.getValue();
				// Write editor's name to the file
				bw.write(editor.getName() + "\n");
				// Then, write his/her alignment to file
				copyAlignmentToFile(bw, alignment);
			}
		}
		catch (FileNotFoundException fnf) {
			System.out.println("File not found error occurred while writing alignment.");
			fnf.printStackTrace();
		} 
		catch (IOException ioe) {
			System.out.println("IOexception occurred while writing alignment.");
			ioe.printStackTrace();
		}	 
	}

	/**
	 * Method which allows team leaders to write the scores of the optimal as well as
	 * all editors' alignments stored in the Repository to a file.
	 * 
	 * @param leader The team leader writing the reports. 
	 */
	public void writeReport(TeamLead leader) {
		// Get the correct path pointing to the desired file
		String currentPath = reportsDirectory.getAbsolutePath();
		File path = new File(currentPath, leader.getFileString(getReportExtension()));
		String fileName = path.getAbsolutePath();
		// Show who's going to write a report to the screen
		System.out.println(leader.getName() + " saves report of team to " + fileName);
		// Write down the scores
		try(BufferedWriter bw = new BufferedWriter(new FileWriter(fileName))){
			// Start off with the score of the optimal alignment
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
			System.out.println("File not found when writing alignment.");
			fnf.printStackTrace();
		} 
		catch (IOException ioe) {
			System.out.println("IOexception when writing alignment.");
			ioe.printStackTrace();
		} 
	}

	/* Getters and setters */

	public HashMap<Editor, StandardAlignment> getTeamAlignments() {
		return this.teamAlignments;
	}

	/* Getters and setters */ 

	public int getOptimalScore() {
		return optimalScore;
	}
	
	/**
	 * The setter for the optimal score is private, as we don't allow users to externally alter
	 * this variable, and moreover checks whether 
	 * @param optimalScore
	 */
	private void setOptimalScore(int optimalScore) {
		if (optimalScore < 0) {
			System.out.println("Difference score should be positive. Exiting method.");
			return;
		}
		this.optimalScore = optimalScore;
	}

	public void updateOptimalScore() {
		int score = this.optimalSA.getDifferenceScore();
		this.setOptimalScore(score);
	}

	// Note!!! These getters are private and hence can only used by repository, not by employees!
	private StandardAlignment getOptimalSA() {
		return optimalSA;
	}
	
	private SNiPAlignment getOptimalSNiP() {
		return optimalSNiP.copy();
	}

	/**
	 * Setter for the standard view of the optimal alignment. This method
	 * automatically constructs a SNiP view as well, and updates the score. 
	 * 
	 * @param optimalSA The new optimal standard alignment to be stored in the repository.
	 */
	private void setOptimalSA(StandardAlignment optimalSA) {
		// Updates the optimal SA. Automatically updates the SNiP and optimal score
		this.optimalSA = optimalSA.copy();

		// Create a SNiP alignment from this SA, and update instance variable
		SNiPAlignment snip = new SNiPAlignment(optimalSA);
		this.setOptimalSNiP(snip);
		this.updateOptimalScore();
	}

	/**
	 * Setter for the SNiP view of the optimal alignment. This method
	 * automatically and updates the difference score. 
	 * 
	 * @param optimalSA The new optimal standard alignment to be stored in the repository.
	 */
	public void setOptimalSNiP(SNiPAlignment optimalSNiP) {
		// Reset the current optimal SNiP alignment based on current optimal SA
		this.optimalSNiP = optimalSNiP.copy();
		// Also update the referenceGenome and referenceID
		this.referenceGenome = this.optimalSNiP.getReferenceGenome();
		this.referenceID     = this.optimalSNiP.getReferenceID();
		// Update optimal score after updating optimal alignment
		this.updateOptimalScore();
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

	public File getReportsDirectory() {
		return reportsDirectory;
	}

	public File getBackupsDirectory() {
		return backupsDirectory;
	}

	public static String getBackupExtension() {
		return BACKUP_EXTENSION;
	}

	public static String getDataExtension() {
		return DATA_EXTENSION;
	}

	public static String getReportExtension() {
		return REPORT_EXTENSION;
	}
	
	/* Visualization (extras) */

	public void displayOptimalSA() {
		this.optimalSA.display();
	}

	public void displayOptimalSNiP() {
		this.optimalSNiP.display();
	}
}
