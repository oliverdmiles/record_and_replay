#include <fstream>
#include <iostream>
#include <set>
#include <unordered_map>

using namespace std;

struct Access {
	uint64_t timestamp;
	uint64_t address;
	int thread_id;
	bool load;
	bool shared;
	double value; // may need to be changed?
};

ostream& operator<<(ostream &os, const Access& a) {
	os << a.timestamp << " " << a.thread_id;
	if (a.load) os << " L ";
	else os << " S ";

	if (a.shared) os << "Sh ";
	else os << "Gl ";

	os << a.value;
	return os;
}

class Acc_compare {
	public:
		// sorting reversed
		bool operator()(const Access *lhs, const Access *rhs) {
			return lhs->timestamp >= rhs->timestamp;
		}
};

// todo: add map for different functions
unordered_map<uint64_t, multiset<Access*, Acc_compare> > accs;

void read_file(char *filename) {
	ifstream file(filename);
	if (!file.is_open()) {
		cout << "Error opening file: " << filename << endl;
	}

	string function;
	uint64_t timestamp;
	string op;
	string type;

	getline(file, function);

	while (true) {
		while (file >> timestamp) {
			Access *acc = new Access;
			acc->timestamp = timestamp;
			file >> acc->address >> acc->thread_id >> op >> type >> acc->value;
			acc->load = (op == "L");
			acc->shared = (type == "S");

			accs[acc->address].insert(acc);
		}
		if (file.fail()) {
			cout << "Reading timestamp failed" << endl;
			file.clear();
			getline(file, function);
			// file in bad state if at end
			cout << function;
			if (!file) return; // TODO: code for multiple functions
		}
	}

	return;
}

void find_dependencies() {
	set<uint64_t> dependencies;
	for (auto map_it = accs.begin(); map_it != accs.end(); map_it++) {
		auto end = map_it->second.end();
		bool read = false;
		bool write = false;
		for (auto it = map_it->second.begin(); it != end; it++) {
			Access *curr = *it;
			read = read || curr->load;
			write = write || !curr->load;
			if (read & write) {
				dependencies.insert(map_it->first);
				break;
			}
		}
	}
	cout << dependencies.size() << endl;

	for (auto adds = dependencies.begin(); adds != dependencies.end(); adds++) {
		cout << *adds;
		auto end = accs[*adds].rend();
		for (auto it = accs[*adds].rbegin(); it != end; it++) {
			cout << " " << (*it)->thread_id;
		}
		cout << endl;
	}

	/*for (auto map_it = accs.begin(); map_it != accs.end(); map_it++) {
		auto end = map_it->second.end();
		//bool print = map_it->second.size() > 1;
		string reads = "";
		string writes = "";
		for (auto it = map_it->second.begin(); it != end; it++) {
			Access *curr = *it;

			if (curr->load) {
				if (writes != "") {
					cout << "address: " << map_it->first << endl;
					cout << "\tread(s): " << reads << endl << "\twrite(s): " << writes << endl;
					reads = writes = "";
				}
				reads += to_string(curr->thread_id) + " " + to_string(curr->value) + " ";
			} else if (reads != "") {
				writes += to_string(curr->thread_id) + " " + to_string(curr->value) + " ";
			}
		}
		if (reads != "" && writes != "") {
			cout << "address: " << map_it->first << endl;
			cout << "\tread(s): " << reads << endl << "\twrite(s): " << writes << endl;
			reads = writes = "";
		}

	}*/
	return;
}

int main(int argc, char *argv[]) {
	if (argc != 2) {
		cout << "Expecting filename as argument" << endl;
	}

	read_file(argv[1]);
	find_dependencies();

	return 0;
}
