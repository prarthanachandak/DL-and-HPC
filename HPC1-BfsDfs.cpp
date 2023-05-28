#include<iostream>
#include<queue>
#include<stack>
#include<omp.h>

using namespace std;

struct TreeNode{
	int data;
	TreeNode* left;
	TreeNode* right;
};

TreeNode* createNode(int data){
	TreeNode* newNode = new TreeNode();
	if(!newNode){
		cout<<"Please try again.";
		return NULL;
	}
	newNode->data = data;
	newNode->left=NULL;
	newNode->right=NULL;
	return newNode;
}

bool ParallelBFS(TreeNode* root, int target){
	if(root==NULL){
		return false;
	}
	queue<TreeNode*> q; //queue pointing to nodes of tree

	q.push(root);

	bool found = false;
	

	#pragma omp parallel

	while(!q.empty()&&!found){
		
		int levelSize = q.size();
		int j=1;
		#pragma omp for 
		for(int i = 0; i < levelSize; i++)
		{
			#pragma omp critical
			TreeNode* current = q.front();
			q.pop();

			if(current->data==target){
				found=true;
				break;
			}

			if(current->left){
				#pragma omp critical
				q.push(current->left);
			}
			if(current->right){
				#pragma omp critical
				q.push(current->right);
			}
		}
	}
	return found;

}

bool ParallelDFS(TreeNode* root, int target){
	if(root==NULL){
		return false;
	}
	stack<TreeNode*> s;

	s.push(root);
	bool found = false;

	#pragma omp parallel
	while(!s.empty() && !found){
		#pragma omp critical
		if(!s.empty()){
			TreeNode* currentNode = s.top();
			s.pop();

			if(currentNode->data==target){
			found=true;
			break;
			}

			if(currentNode->right){
				#pragma omp critical
				s.push(currentNode->right);
			}

			if(currentNode->left){
				#pragma omp critical
				s.push(currentNode->left);
			}
		}

	}
	return found;
}

int main(){
	TreeNode* root = createNode(1);
	root->left = createNode(2);
	root->right = createNode(3);
	root->left->left = createNode(4);
	root->left->right = createNode(5);
	root->right->right = createNode(6);

	int n;
	cin>>n;
	if(ParallelBFS(root, n)){
		cout<<"found!\n";
	}
	else{
		cout<<"not found!\n";
	}

	if(ParallelDFS(root, n)){
		cout<<"found!\n";
	}
	else{
		cout<<"not found!\n";
	}
}