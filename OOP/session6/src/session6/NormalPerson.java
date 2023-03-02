package session6;

public class NormalPerson {

	private String name;
	private String gender;
	
	public NormalPerson(String name, String gender) {
		this.setName(name);
		this.setGender(gender);
	}

	
	// Getters and setters
	public void setName(String newName) {
		this.name = newName;
	}
	
	public String getName() {
		return this.name;
	}
	
	public void setGender(String newGender) {
		this.gender = newGender;
	}

	public String getGender() {
		return this.gender;
	}
}
