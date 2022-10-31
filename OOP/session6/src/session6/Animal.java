package session6;

public abstract class Animal implements Feedable {
	
	// By default, created animals are hungry
	private final String name;
	private final String product;
	private boolean hungry = true;
	private int age;
	
	public Animal(String name, int age, String product) {
		this.name = name;
		this.age = age;
		this.product = product;
	}
	
	@Override
	public void feed() {
		this.hungry = false;
	}
	
	@Override
	public boolean isHungry() {
		return this.hungry;
	}
	
	// Birthday: increase age by 1
	public void birthday() {
		System.out.println("It's "  + this.getName() + " birthday!");
		this.age += 1;
	}
	
	// Getters and setters
	public String getName() {
		return this.name;
	}
	
	public int getAge() {
		return this.age;
	}
	
	
}
