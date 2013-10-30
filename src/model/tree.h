//////////////////////////////////
//
// == A reuse distance based GPU cache model
// This file is part of a cache model for GPUs. The cache model is based on
// reuse distance theory extended to work with GPUs. The cache model primarly
// focusses on modelling NVIDIA's Fermi architecture.
//
// == More information on the GPU cache model
// Article............A Detailed GPU Cache Model Based on Reuse Distance Theory
// Authors............C. Nugteren et al.
//
// == Contents of this file
// This particular file implements a tree structure to be used with an efficient
// implementation of the reuse distance theory (Bennett and Kruskal's Algorithm)
// as presented in literature (see below - section 4.4). The tree structure is a
// partial sum-hierarchy tree. It is chosen for its complexity of implementation
// versus performance trade-off. Other trees could perform better, but change
// over time, requiring additional implementation effort.
//
// == More information on reuse distance implementation
// Article............Calculating stack distances efficiently
// Authors............George Almasi, Calin Cascaval, and David Padua
// DOI................http://dx.doi.org/10.1145/773146.773043
//
// == File details
// Filename...........src/model/tree.h
// Author.............Cedric Nugteren <www.cedricnugteren.nl>
// Affiliation........Eindhoven University of Technology, The Netherlands
// Last modified on...22-Jul-2013
//
//////////////////////////////////

#ifndef TREE_H
#define TREE_H

//////////////////////////////////
// Floor and ceiling functions
//////////////////////////////////
#define CEIL_DIV(a,b) (float)(a+b-1)/(float)b
#define FLOOR_DIV(a,b) (float)a/(float)b

//////////////////////////////////
// The tree-node for a partial sum-hierarchy tree
//////////////////////////////////
class Node {
public:
	Node* left;                   // Pointer to the node on the left
	Node* right;                  // Pointer to the node on the right
	unsigned range_b;             // Indicator of the range of nodes on the right
	unsigned value;               // Value of the node: 0 or 1 if it is a leaf, 0+ otherwise
	
	// Constructor
	Node(unsigned _b, unsigned _value) {
		range_b = _b;
		value = _value;
		left = 0;
		right = 0;
	}
};

//////////////////////////////////
// A partial sum-hierarchy tree
//////////////////////////////////
class Tree {
public:
	Node* root;
	
	// Initialize the tree and fill it with a given size
	Tree(unsigned _size) {
		root = fill_tree(0,_size,0);
	}
	
	// Delete the tree
	~Tree() {
		clean_tree(root);
	}

	// Method to recursively fill the tree with nodes
	Node* fill_tree(unsigned start, unsigned size, unsigned level) {
		Node* node = new Node(start+size-1,0);
		if (size > 1) {
			unsigned val_left = CEIL_DIV(size,2);
			unsigned val_right = FLOOR_DIV(size,2);
			node->left = fill_tree(start,         val_left, level+1);
			node->right = fill_tree(start+val_left,val_right,level+1);
		}
		return node;
	}
	
	// Method to recursively delete the nodes in the tree
	void clean_tree(Node* node) {
		if (node->left  != 0) { clean_tree(node->left ); }
		if (node->right != 0) { clean_tree(node->right); }
		delete node;
	}

	// Count all values right of a given node (the target)
	unsigned count(unsigned target) {
		unsigned result = 0;
		Node* node = root;
		
		// Reached the leaf or found an empty sub-tree
		while (node->left != 0 && node->value != 0) {
			// Go right...
			if (target > node->left->range_b) {
				node = node->right;
			}
			// ...or go left
			else {
				result += node->right->value;
				node = node->left;
			}
		}
		return result;
	}
	
	// Set a given node's value to 1
	void set(unsigned target) {
		Node* node = root;
		
		// Iterate until the leaf is found
		while (node->left != 0) {
			node->value += 1;
			// Go right...
			if (target > node->left->range_b) {
				node = node->right;
			}
			// ...or go left
			else {
				node = node->left;
			}
		}
		node->value = 1;
	}
	
	// Set a given node's value to 0
	void unset(unsigned target) {
		Node* node = root;
		
		// Iterate until the leaf is found
		while (node->left != 0) {
			node->value -= 1;
			// Go right...
			if (target > node->left->range_b) {
				node = node->right;
			}
			// ...or go left
			else {
				node = node->left;
			}
		}
		node->value = 0;
	}
};

//////////////////////////////////

#endif