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
unordered_map<string, int> filenames;

void find_dependencies(string);

string get_filename(string function) {
	auto split = function.find_last_of(" ");
	string output_file = function.substr(split + 1);
	if (filenames.find(output_file) != filenames.end()) {
		filenames[output_file]++;
	} else {
		filenames[output_file] = 1;
	}
	output_file += "_" + to_string(filenames[output_file] - 1) + ".txt";
	return output_file;
}

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
	string output_file = get_filename(function);

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
			find_dependencies(output_file);
			cout << "Reading timestamp failed" << endl;
			file.clear();
			getline(file, function);
			if (!file) return; // Have reached end of file
			output_file = get_filename(function);
			accs.clear();
		}
	}

	return;
}

void find_dependencies(string output_file) {
	ofstream file(output_file);
	if (!file.is_open()) {
		cout << "Error opening output file " << output_file << endl;
	}

	set<Access*, Acc_compare> dependencies;
	for (auto map_it = accs.begin(); map_it != accs.end(); map_it++) {
		auto end = map_it->second.end();
		bool read = false;
		bool write = false;
		for (auto it = map_it->second.begin(); it != end; it++) {
			Access *curr = *it;
			read = read || curr->load;
			write = write || !curr->load;
			if (read & write) {
				dependencies.insert(*map_it->second.begin());
				break;
			}
		}
	}
	file << dependencies.size() << endl;

	for (auto acc = dependencies.begin(); acc != dependencies.end(); acc++) {
		uint64_t addr = (*acc)->address;
		file << addr << " " << accs[addr].size();
		auto end = accs[addr].rend();
		for (auto it = accs[addr].rbegin(); it != end; it++) {
			file << " " << (*it)->thread_id;
		}
		file << endl;
	}

	return;
}

int main(int argc, char *argv[]) {
	if (argc != 2) {
		cout << "Expecting filename as argument" << endl;
		exit(1);
	}

	read_file(argv[1]);

	return 0;
}
