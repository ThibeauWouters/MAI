package Team;

//import java.io.BufferedWriter;
//import java.io.FileNotFoundException;
//import java.io.FileWriter;
//import java.io.IOException;
import java.time.LocalDateTime;

import Alignments.*;

public class TechnicalSupport extends Employee {
	
	/* Variables */
	private String backupTimestamp = "No backup created yet.";
	
	/* Constructor */
	
	public TechnicalSupport(String firstName, String lastName, int yearsOfExperience) {
		super("TechnicalSupport", firstName, lastName, yearsOfExperience);
	}
	
	/* Methods */
	
	public void createBackup(Repository repo) {
		// Makes a backup of the corresponding repo
		
		// Get a timestamp when creating the backup. Store as instance variable.
		String timeStamp = getDateAndTime();
		this.backupTimestamp = timeStamp;
		
		// Create the fileName

		String folderName = "src/Backups/";
		String fileName = folderName + this.getFirstName() + "_" + this.getLastName() + ".backup.txt";
		System.out.println("Saving backup to " + fileName);
		
		// Do the backup in Repository class
		repo.backup(this, fileName);
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
		this.backupTimestamp = backupTimestamp;
	}
	
	public void displayBackupTimestamp() {
		System.out.println(this.getName() + " has made his/her last backup at " + this.getBackupTimestamp());
	}

	public static String getDateAndTime() {
		// Finds and returns the current date and time as a String
		return LocalDateTime.now().toString();
	}
}
