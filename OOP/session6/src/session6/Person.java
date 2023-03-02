package session6;

public class Person implements PersonalizedPrint, CanSing{
	
	//// Not entirely what the idea was, see solutions --- boolean canSing no longer needed
	private String personPrint;
	private boolean canSing;
	
	public Person(String personPrint, boolean canSing) {
		this.setPersonPrint(personPrint);
		this.setCanSing(canSing);
	}
	
	// CanSing method
	
	@Override
	public void sing() {
		System.out.println("laaaalaaalaaa");
	}
	
	@Override
	public void makeNoiseOne() {
		System.out.println("eeeekeeekeeeek");;
	}
	
	@Override
	public void makeNoiseTwo() {
		System.out.println("ueueueueueueue");
	}
	
	public String prettyPrint() {
		return this.personPrint;
	}
	
	// Getters and setters
	public String getPersonPrint() {
		return this.personPrint;
	}
	
	public void setPersonPrint(String personPrint) {
		this.personPrint = personPrint;
	}
	
	public void setCanSing(boolean canSing) {
		this.canSing = canSing;
	}
	
	// Getters and setters
	public boolean getCanSing() {
		return this.canSing;
	}
}
