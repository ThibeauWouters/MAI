package team;

import java.util.ArrayList;

//import alignments.*;
import team.editors.Bioinformatician;
import team.editors.Editor;

/**
 * This file holds all team employees and can implement all future functionalities related to the team in general.
 *
 * @author ThibeauWouters
 *
 */
public class Team {

	/**
	 * team is an ArrayList holding all Employee objects belonging to this team.
	 * An ArrayList is flexible, meaning future applications can easily
	 * perform operations such as adding or removing team members.
	 */
	ArrayList<Employee> team;
	
	/* Constructor */
	
	/**
	 * The constructor directly takes an existing ArrayList of employees and stores it as instance variable.
	 * 
	 * @param teamArrayList An ArrayLists holding Employee objects of a new team to be created.
	 */
	public Team(ArrayList<Employee> teamArrayList) {
		this.team = teamArrayList;
	}

	/* Methods */
	
	public void removeEmployee(Employee employee) {
		// Remove an employee from the team
		this.team.remove(employee);
	}
	
	public void addEmployee(Employee employee) {
		// Add an employee to the team
		this.team.add(employee);
	}
	
	/* Getters and setters */
	
	public ArrayList<Employee> getTeam(){
		return this.team;
	}
	
	/**
	 * Auxiliary method showing the current team members in an organized manner.
	 * Used just for showing the operation of the application in the main method.
	 */
	public void displayTeamMembers() {
		for (Employee employee : this.getTeam()) {
			System.out.println("Employee name:    " + employee.getName());
			System.out.println("Employee type:    " + employee.getEmployeeType());
			System.out.println("Experience (yrs): " + employee.getYearsOfExperience());
		}
	}
	
	/* Getters to return lists for specific employee types */
	
	// Note: the methods below are mainly for illustration purposes in the main method,
	// and may not be useful outside their application in the main method!!!
	
	public ArrayList<Editor> getEditors(){
		// Create empty list
		ArrayList<Editor> result = new ArrayList<Editor>();
		// Fill it: iterate over members, save the editors:
		for (Employee employee : this.getTeam()) {
			if (employee instanceof Editor) {
				Editor editor = (Editor) employee;
				result.add(editor);
			}
		}
		return result;
	}
	
	public ArrayList<Bioinformatician> getBioinformaticians(){
		// Create empty list
		ArrayList<Bioinformatician> result = new ArrayList<Bioinformatician>();
		// Fill it: iterate over members, save the editors:
		for (Employee employee : this.getTeam()) {
			if (employee instanceof Bioinformatician) {
				Bioinformatician editor = (Bioinformatician) employee;
				result.add(editor);
			}
		}
		return result;
	}
	
	public ArrayList<TeamLead> getTeamLeads(){
		// Create empty list
		ArrayList<TeamLead> result = new ArrayList<TeamLead>();
		// Fill it: iterate over members, save the editors:
		for (Employee employee : this.getTeam()) {
			if (employee instanceof TeamLead) {
				TeamLead editor = (TeamLead) employee;
				result.add(editor);
			}
		}
		return result;
	}
	
	public ArrayList<TechnicalSupport> getTechnicalSupports(){
		// Create empty list
		ArrayList<TechnicalSupport> result = new ArrayList<TechnicalSupport>();
		// Fill it: iterate over members, save the editors:
		for (Employee employee : this.getTeam()) {
			if (employee instanceof TechnicalSupport) {
				TechnicalSupport editor = (TechnicalSupport) employee;
				result.add(editor);
			}
		}
		return result;
	}
}
