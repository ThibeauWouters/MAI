package team;

import java.time.LocalDateTime;
import datastructures.*;
import team.editors.Editor;

public class TechnicalSupport extends Employee {
	
	/* Variables */
	private String backupTimestamp;
	
	/* Constructor */
	
	public TechnicalSupport(String firstName, String lastName, int yearsOfExperience) {
		super("TechnicalSupport", firstName, lastName, yearsOfExperience);
		this.setHasRepositoryAccess(true);
	}
	
	/* Methods */
	
	/* Creating a backup */
	
	public void createBackupOptimal(Repository repo) {
		// Makes a backup of the corresponding repo
		
		// Get a timestamp when creating the backup. Store as instance variable.
		this.setBackupTimestamp(getDateAndTime());
		// Do the backup in Repository class
		repo.backupOptimal(this);
	}
	
	public void createBackup(Repository repo, Editor editor) {
		// Makes a backup of the corresponding repo
		
		// Get a timestamp when creating the backup. Store as instance variable.
		this.setBackupTimestamp(getDateAndTime());
		// Do the backup in Repository class
		repo.backup(this, editor);
	}
	
	public void createBackupEditors(Repository repo) {
		// Makes a backup of the corresponding repo
		
		// Get a timestamp when creating the backup. Store as instance variable.
		this.setBackupTimestamp(getDateAndTime());
		// Do the backup in Repository class
		repo.backupEditors(this);
	}
	
	public void createBackup(Repository repo) {
		// Makes a backup of the corresponding repo
		
		// Get a timestamp when creating the backup. Store as instance variable.
		this.setBackupTimestamp(getDateAndTime());
		// Do the backup in Repository class
		repo.backup(this);
	}
	
	/* Loading backup */
	
	public void load(Repository repo) {
		repo.load(this);
	}
	
	public void load(Repository repo, Editor editor) {
		repo.load(this, editor);
	}
	
	public void loadEditors(Repository repo) {
		repo.loadEditors(this);
	}
	
	public void loadOptimalSA(Repository repo) {
		repo.loadOptimalSA(this);
	}
	
	/* Clearing repository */
	
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

	private static String getDateAndTime() {
		// Finds and returns the current date and time as a String
		return LocalDateTime.now().toString();
	}
}
