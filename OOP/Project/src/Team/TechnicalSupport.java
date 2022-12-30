package team;

import java.time.LocalDateTime;
import datastructures.*;
import team.editors.Editor;

public class TechnicalSupport extends Employee {
	
	/* Variables */
	private String backupTimestamp;
	private String backupString;
	
	/* Constructor */
	
	public TechnicalSupport(String firstName, String lastName, int yearsOfExperience) {
		super("TechnicalSupport", firstName, lastName, yearsOfExperience);
	}
	
	/* Methods */
	
	public void createBackup(Repository repo) {
		// Makes a backup of the corresponding repo
		
		// Get a timestamp when creating the backup. Store as instance variable.
		String timeStamp = getDateAndTime();
		this.setBackupTimestamp(timeStamp);
		
		// Create the fileName
		String fileName = this.getFirstName() + "_" + this.getLastName() + ".backup.txt";
		System.out.println("Saving backup to " + fileName);
		
		// Do the backup in Repository class
		repo.backup(this);
	}
	
	public void loadBackup(String fileName, Repository repo) {
		repo.load(this, fileName);
	}
	
	public void clearRepository(Repository repo) {
		repo.clear(this);
	}

	/* Getters and setters */
	
	public String getBackupTimestamp() {
		return backupTimestamp;
	}
	
	public void setBackupTimestamp(String backupTimestamp) {
		// Update the backupTimeStamp
		this.backupTimestamp = backupTimestamp;
		// Update the backup string, which uses backupTimeStamp
		this.updateBackupString();
	}
	
	private void updateBackupString() {
		// If the backupTimeStamp is changed, this method is called, and changes the backup string
		this.backupString = this.getFirstName() + "_" + this.getLastName() + ".backup.";
	}

	public void displayBackupTimestamp() {
		if (this.backupString == null) {
			System.out.println(this.getName() + " has not made backups yet.");
		} else {
			System.out.println(this.getName() + " has made his/her last backup at " + this.getBackupTimestamp());
		}
		
	}

	public static String getDateAndTime() {
		// Finds and returns the current date and time as a String
		return LocalDateTime.now().toString();
	}

	public String getBackupString() {
		return backupString;
	}
	
	/* Methods to create fileNames to create backup files */
	
	public String getBackupString(Editor editor) {
		return backupString + editor.getFirstName() + "_" + editor.getLastName() + ".txt";
	}
	
	public String getBackupString(String str) {
		return backupString + str + ".txt";
	}
}
