package controllers.miniAlphaZero.simulator.mtd;


import ontology.Types;

public class ActionValuePair implements Comparable<ActionValuePair> {

	Types.ACTIONS action;
	ActionValuePair principalVariation;
	int value, previousValue;
	
	public ActionValuePair(Types.ACTIONS a, int v) {
		this.action = a;
		this.value = v;
		this.previousValue = 0;
		this.principalVariation = null;
	}

	@Override
	public int compareTo(ActionValuePair other) {
		return Float.compare(this.value, other.value);
	}
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("(Action : ");
		sb.append(action);
		/*
		ActionValuePair pv = this.principalVariation;
		while (pv.action != null) {
			sb.append("->");
			sb.append(pv.action);
			pv = pv.principalVariation;
		}
		*/
		sb.append(" Value: " + value + ")");
		return sb.toString();
	}
	
}
