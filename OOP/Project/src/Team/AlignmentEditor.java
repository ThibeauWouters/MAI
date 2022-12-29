package Team;

//import java.io.BufferedWriter;
//import java.io.FileNotFoundException;
//import java.io.FileWriter;
//import java.io.IOException;

import Alignments.Repository;
import Alignments.StandardAlignment;

public abstract class AlignmentEditor extends Employee {
	
	/* Variables */
	private StandardAlignment alignment; 
	
	/* Constructor */
	
	public AlignmentEditor(String employeeType, String firstName, String lastName, int yearsOfExperience, StandardAlignment sa) {
		super(employeeType, firstName, lastName, yearsOfExperience);
		this.alignment = sa;
	}
	
	public StandardAlignment getAlignment() {
		return this.alignment;
	}
	
	public void setAlignment(StandardAlignment sa) {
		this.alignment = sa;
	}
	
	// TODO - add edit functionalities here for the alignment?
	
	public int getScore() {
		// Return the score of this editor's current alignment
		return this.alignment.getDifferenceScore();
	}
	
	/* Function to write the alignment */
	
	public void writeData(Repository repo) {
		repo.writeData(this);
	}
	
	public void writeReport(Repository repo) {
		repo.writeReport(this);
	}
	
//	public void writeData() {
//		String folderName = "src/";
//		String fileName = folderName + this.getFirstName() + "_" + this.getLastName() + ".alignment.txt";
//		System.out.println("Saving alignment of " + this.getName() + " to " + fileName);
//		
//		try(BufferedWriter bw = new BufferedWriter(new FileWriter(fileName))){
//			
//			for (String identifier : this.getAlignment().getIdentifiers()) {
//				String genome = this.getAlignment().getGenome(identifier);
//				bw.write(identifier + "\n");
//				bw.write(genome + "\n");
//			}
//			
//		}
//		catch (FileNotFoundException fnf) {
//			System.out.println("File not found when writing alignment: " + fnf);
//		} 
//		catch (IOException ioe) {
//			System.out.println("IOexception when writing alignment: " + ioe);
//		} 
//	}
//	
//	public void writeReport() {
//		String folderName = "src/";
//		String fileName = folderName + this.getFirstName() + "_" + this.getLastName() + ".score.txt";
//		System.out.println("Saving score of " + this.getName() + " to " + fileName);
//		
//		try(BufferedWriter bw = new BufferedWriter(new FileWriter(fileName))){
//			int score = this.getScore();
//			bw.write(score + "\n");
//		}
//		catch (FileNotFoundException fnf) {
//			System.out.println("File not found when writing alignment: " + fnf);
//		} 
//		catch (IOException ioe) {
//			System.out.println("IOexception when writing alignment: " + ioe);
//		} 
//	}
}
