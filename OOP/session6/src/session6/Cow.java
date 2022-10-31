package session6;

public class Cow extends Animal {
	
	private boolean milk;
	
	public Cow(String name, int age, boolean milk) {
		super(name, age, "Milk");
		this.milk= milk;
	}
}
