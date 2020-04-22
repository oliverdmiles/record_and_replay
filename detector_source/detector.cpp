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
	uint64_t value;
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
		bool operator()(const Access *lhs, const Access *rhs) {
			return lhs->timestamp < rhs->timestamp;
		}
};

// todo: add map for different functions
unordered_map<uint64_t, multiset<Access*, Acc_compare> > accs;
unordered_map<string, int> filenames;

void find_dependencies(string);

void read_file(string filename) {
	ifstream file(filename);
	if (!file.is_open()) {
		cout << "Error opening file: " << filename << endl;
	}

	uint64_t timestamp;
	string op;
	string type;

	unordered_map<int, multiset<Access*, Acc_compare> > thread_pairs;

	while (file >> timestamp) {
		Access *acc = new Access;
		acc->timestamp = timestamp;
        	string temp;
		file >> acc->address >> acc->thread_id >> op >> type >> temp;
        	acc->value = std::stoull(temp, nullptr, 0);
		acc->load = (op == "L");
		acc->shared = (type == "S");

		if (acc->load) {
			thread_pairs[acc->thread_id].insert(acc);
		} else {
			accs[acc->address].insert(acc);
		}

	}

	for (auto map_it = thread_pairs.begin(); map_it != thread_pairs.end(); map_it++) {
		for (auto it = map_it->second.begin(); it != map_it->second.end(); it++) {
			Access* temp = *it;
			it++;
			Access* just_value = *it;

			temp->value = just_value->value;
			accs[temp->address].insert(temp);
		}
		map_it->second.clear();
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
		//if (map_it->second.size() > 1) {
			dependencies.insert(*map_it->second.begin());
		//}
	}
	file << dependencies.size() << endl;

	for (auto acc = dependencies.begin(); acc != dependencies.end(); acc++) {
		uint64_t addr = (*acc)->address;
		file << addr << " " << accs[addr].size();
		auto end = accs[addr].end();
		for (auto it = accs[addr].begin(); it != end; it++) {
			file << " " << (*it)->thread_id << " " << ((*it)->load ? "L" : "S");
			file << " 0x" << std::hex << (*it)->value << std::dec;
		}
		file << endl;
	}

	return;
}

void clear_map() {
	for (auto map_it = accs.begin(); map_it != accs.end(); map_it++) {
		for (auto it = map_it->second.begin(); it != map_it->second.end(); it++) {
			Access* temp = *it;
			delete temp;
		}
	}
	accs.clear();
	return;
}


int main(int argc, char *argv[]) {
	if (argc < 2) {
		cout << "Expecting filename(s) as argument" << endl;
		exit(1);
	}

	for (int i = 1; i < argc; i++) {
		string infile = argv[i];
		if (infile.substr(infile.size() - 7) != ".record") {
			cout << "Filename must end with \".record\"" << endl;
			continue;
		}

		read_file(infile);

		/*for (auto i = accs.begin(); i != accs.end(); i++) {
			cout << i->first;
			auto it = i->second.begin();
			auto end = i->second.end();
			for (; it != end; it++) {
				cout << *(*it) << " ";
			}
			cout << endl;
		}*/
		
		auto begin = infile.find_last_of("/");
		auto end = infile.find_last_of(".");
		string outfile = "dependency_output/" + infile.substr(begin + 1, end - begin - 1) + ".dependencies";
		find_dependencies(outfile);
		clear_map();
	}

	return 0;
}
