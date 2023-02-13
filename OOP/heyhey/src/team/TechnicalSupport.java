package team;

import java.time.LocalDateTime;
import datastructures.*;
import team.editors.Editor;

/**
 * This class implements all the functionalities of the technical staff member.
 */
public class TechnicalSupport extends Employee {
	
	/* Variables */

	/**
	 * backupTimestamp shows the last date and time when a specific technical staff member made a backup of the
	 * repository
	 */
	private String backupTimestamp;
	
	/* Constructor */

	/**
	 * Use the employee constructor here
	 *
	 * @param firstName The first name of this employee
	 * @param lastName The last name of this employee
	 * @param yearsOfExperience The number of years of experience of this employee
	 */
	public TechnicalSupport(String firstName, String lastName, int yearsOfExperience) {
		super("TechnicalSupport", firstName, lastName, yearsOfExperience);
	}
	
	/* Methods */
	
	/* Creating a backup */

	/**
	 * Create a backup of the optimal alignment of a repository.
	 *
	 * @param repo Repository of which we are going to make a backup of its optimal alignment
	 */
	public void createBackupOptimal(Repository repo) {
		// Get a timestamp when creating the backup. Store as instance variable.
		this.setBackupTimestamp(getDateAndTime());
		// Do the backup in Repository class
		repo.backupOptimal(this);
	}

	/**
	 * Create a backup of the alignment of a single editor in a repository.
	 *
	 * @param repo Repository in which we are going to make a backup of the editor
	 * @param editor The editor of which we are going to make a backup the alignment in the repository
	 */
	public void createBackup(Repository repo, Editor editor) {
		// Get a timestamp when creating the backup. Store as instance variable.
		this.setBackupTimestamp(getDateAndTime());
		// Do the backup in Repository class
		repo.backup(this, editor);
	}

	/**
	 * Create a backup of all alignment of editors working in a repository.
	 *
	 * @param repo Repository in which we are going to make a backup of all its editors
	 */
	public void createBackupEditors(Repository repo) {
		// Get a timestamp when creating the backup. Store as instance variable.
		this.setBackupTimestamp(getDateAndTime());
		// Do the backup in Repository class
		repo.backupEditors(this);
	}

	/**
	 * Create a full backup (optimal alignment and all editor alignments) of a repository
	 *
	 * @param repo Repository in which we are going to make the backup
	 */
	public void createBackup(Repository repo) {
		// Get a timestamp when creating the backup. Store as instance variable.
		this.setBackupTimestamp(getDateAndTime());
		// Do the backup in Repository class
		repo.backup(this);
	}
	
	/* Loading backup */

	/**
	 * Load the backup files of a repository
	 *
	 * @param repo Repository of which we are going to load the backup
	 */
	public void load(Repository repo) {
		repo.load(this);
	}

	/**
	 * Load the backup file of a single editor of a repository
	 *
	 * @param repo Repository of which we are going to load the backup
	 * @param editor Editor of which we are going to load the backup
	 */
	public void load(Repository repo, Editor editor) {
		repo.load(this, editor);
	}

	/**
	 * Load the backup file of all editors of a repository
	 *
	 * @param repo Repository of which we are going to load the backup
	 */
	public void loadEditors(Repository repo) {
		repo.loadEditors(this);
	}

	/**
	 * Load the backup file of the optimal alignment of a repository
	 *
	 * @param repo Repository of which we are going to load the backup
	 */
	public void loadOptimalSA(Repository repo) {
		repo.loadOptimalSA(this);
	}
	
	/* Clearing repository */

	/**
	 * Clear all the contents of a repository (i.e., all its alignments stored as instance variables)
	 *
	 * @param repo Repository which e wish to celar
	 */
	public void clear(Repository repo) {
		repo.clear(this);
	}

	/* Getters and setters */
	
	public String getBackupTimestamp() {
		return backupTimestamp;
	}
	
	public void setBackupTimestamp(String backupTimestamp) {
		this.backupTimestamp = backupTimestamp;
	}

	public void displayBackupTimestamp() {
		if (this.backupTimestamp == null) {
			System.out.println(this.getName() + " has not made backups yet.");
		} else {
			System.out.println(this.getName() + " has made their last backup at " + this.getBackupTimestamp());
		}
		
	}

	/**
	 * Auxiliary method to create a unique timestamp, used when creating backups.
	 *
	 * @return String displaying the date and time at which this method is called.
	 */
	private static String getDateAndTime() {
		// Finds and returns the current date and time as a String
		return LocalDateTime.now().toString();
	}
}
