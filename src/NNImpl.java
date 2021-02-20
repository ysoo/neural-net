/**
 * The main class that handles the entire network
 * Has multiple attributes each with its own use
 * 
 */

import java.util.*;


public class NNImpl{
	public ArrayList<Node> inputNodes=null;//list of the input layer nodes.
	public ArrayList<Node> hiddenNodes=null;//list of the hidden layer nodes
	public ArrayList<Node> outputNodes=null;// list of the output layer nodes

	public ArrayList<Instance> trainingSet=null;//the training set

	Double learningRate=1.0; // variable to store the learning rate
	int maxEpoch=1; // variable to store the maximum number of epochs

	/**
	 * This constructor creates the nodes necessary for the neural network
	 * Also connects the nodes of different layers
	 * After calling the constructor the last node of both inputNodes and  
	 * hiddenNodes will be bias nodes. 
	 */

	public NNImpl(ArrayList<Instance> trainingSet, int hiddenNodeCount, Double learningRate, int maxEpoch, Double [][]hiddenWeights, Double[][] outputWeights)
	{
		this.trainingSet=trainingSet;
		this.learningRate=learningRate;
		this.maxEpoch=maxEpoch;

		//input layer nodes
		inputNodes=new ArrayList<Node>();
		int inputNodeCount=trainingSet.get(0).attributes.size();
		int outputNodeCount=trainingSet.get(0).classValues.size();
		for(int i=0;i<inputNodeCount;i++)
		{
			Node node=new Node(0);
			inputNodes.add(node);
		}

		//bias node from input layer to hidden
		Node biasToHidden=new Node(1);
		inputNodes.add(biasToHidden);

		//hidden layer nodes
		hiddenNodes=new ArrayList<Node> ();
		for(int i=0;i<hiddenNodeCount;i++)
		{
			Node node=new Node(2);
			//Connecting hidden layer nodes with input layer nodes
			for(int j=0;j<inputNodes.size();j++)
			{
				NodeWeightPair nwp=new NodeWeightPair(inputNodes.get(j),hiddenWeights[i][j]);
				node.parents.add(nwp);
			}
			hiddenNodes.add(node);
		}

		//bias node from hidden layer to output
		Node biasToOutput=new Node(3);
		hiddenNodes.add(biasToOutput);

		//Output node layer
		outputNodes=new ArrayList<Node> ();
		for(int i=0;i<outputNodeCount;i++)
		{
			Node node=new Node(4);
			//Connecting output layer nodes with hidden layer nodes
			for(int j=0;j<hiddenNodes.size();j++)
			{
				NodeWeightPair nwp=new NodeWeightPair(hiddenNodes.get(j), outputWeights[i][j]);
				node.parents.add(nwp);
			}	
			outputNodes.add(node);
		}	
	}

	/**
	 * Get the output from the neural network for a single instance
	 * Return the index with highest output values. For example if the outputs
	 * of the outputNodes are [0.1, 0.5, 0.2, 0.1, 0.1], it should return 1. If outputs
	 * of the outputNodes are [0.1, 0.5, 0.1, 0.5, 0.2], it should return 3. 
	 * The parameter is a single instance. 
	 */

	public int calculateOutputForInstance(Instance inst)
	{
		//set inputnodes value to attributes
		for(int i = 0; i < inst.attributes.size(); i++) {	// because bias node
			Node curr = inputNodes.get(i);
			double input = inst.attributes.get(i);
			curr.setInput(input);	
		}
		//calculate output of hidden layer
		for(int i = 0; i < hiddenNodes.size(); i ++) {
			Node curr = hiddenNodes.get(i);
			curr.calculateOutput();
		}
		// Output Array
		double[] output = new double[outputNodes.size()];

		//calculate output of output layer
		for(int i = 0 ; i < outputNodes.size(); i ++) {
			Node curr = outputNodes.get(i);
			curr.calculateOutput();
			output[i] = curr.getOutput();
		}

		double max = output[0];
		int returnable = 0;
		for(int i = 0 ; i < output.length; i ++) {
			if(output[i] > max) {
				returnable = i;
			}
		}
		return returnable;
	}

	public Double [] calculateOutput(Instance inst) {
		//set inputnodes value to attributes
		for(int i = 0; i < inst.attributes.size(); i++) {
			Node curr = inputNodes.get(i);
			double input = inst.attributes.get(i);
			curr.setInput(input);	
		}
		//calculate output of hidden layer
		for(int i = 0; i < hiddenNodes.size(); i ++) {
			Node curr = hiddenNodes.get(i);
			curr.calculateOutput();
		}
		// Output Array
		Double[] output = new Double[outputNodes.size()];

		//calculate output of output layer
		for(int i = 0 ; i < outputNodes.size(); i ++) {
			Node curr = outputNodes.get(i);
			curr.calculateOutput();
			output[i] = curr.getOutput();
		}
		return output;
	}

	/**
	 * Train the neural networks with the given parameters
	 * 
	 * The parameters are stored as attributes of this class
	 */

	public void train()
	{
		// repeat until some stopping criterion is met
		for(int m = 0; m < maxEpoch; m++) {
			for(int e = 0 ; e < trainingSet.size(); e++) {

				// for each node in the output layer compute the error
				Double [] prediction = calculateOutput(trainingSet.get(e));
				ArrayList<Integer> actual = trainingSet.get(e).classValues;

				Double[] deltak = new Double[outputNodes.size()];
				Double[] deltaj = new Double[hiddenNodes.size()];

				double[] sumweightk = new double[hiddenNodes.size()];

				// for each hidden unit between hidden and output compute delta k
				for(int k = 0 ; k < outputNodes.size(); k ++) {
					double error = actual.get(k) - prediction[k];	
					double deltak1 = error*prediction[k]*(1 - prediction[k]);
					deltak[k] = deltak1;

					// Populate sumweightk
					Node curr = outputNodes.get(k);
					ArrayList<NodeWeightPair> parents = curr.parents;
					for(int i = 0; i < parents.size(); i ++) {
						NodeWeightPair currnwp = parents.get(i);
						sumweightk[i] = sumweightk[i] + currnwp.getWeight()*deltak1; 
					}
				}

				// for each input unit between input and hidden unit compute delta j
				for(int j = 0; j < hiddenNodes.size() - 1; j++) { // bias node has no deltaj
					Node curr = hiddenNodes.get(j);
					double aj = curr.getOutput();
					double deltaj1 = aj*(1-aj)*sumweightk[j];
					deltaj[j] = deltaj1;
				}

				// Update weights for hidden nodes
				for(int j = 0 ; j < hiddenNodes.size() - 1; j ++) {		//bias node no deltaj
					Node curr = hiddenNodes.get(j);
					ArrayList<NodeWeightPair> parentI = curr.parents;

					for(int i = 0 ;i < parentI.size(); i ++) {
						NodeWeightPair hNWP = parentI.get(i);
						Node I = hNWP.getNode();
						double ai = I.getOutput();
						double deltaweightij = learningRate*ai*deltaj[j];
						double weight = hNWP.getWeight() + deltaweightij;
						hNWP.setWeight(weight);
					}
				}
				// Update weights for output nodes
				for(int k = 0 ; k < outputNodes.size(); k ++) {
					Node curr = outputNodes.get(k);
					ArrayList<NodeWeightPair> parentH = curr.parents;

					for(int i = 0 ;i < parentH.size(); i ++) {
						NodeWeightPair hNWP = parentH.get(i);
						Node H = hNWP.getNode();
						double aj = H.getOutput();
						double deltaweightjk = learningRate*aj*deltak[k];
						double weight = hNWP.getWeight() + deltaweightjk;
						hNWP.setWeight(weight);
					}
				}
			}
		}
	}
}
