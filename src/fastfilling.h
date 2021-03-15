#include <iostream>
#include <vector>
// unsigned int fillTH1Fast1(const std::vector<float> & corrArr) {
// std::cout << "SIZE: " << corrArr.size() << std::endl;
// return corrArr.size();
// }
// 





class HistoFiller {
public:
  HistoFiller(TH1F *histo, const std::vector<float> & source, const std::vector<bool> & filter) :
  histo(histo),
  filter(filter),
  source(source) {
    // std::cout << "Add histo: " << histo->GetName() << ": " << histo << std::endl;
    // std::cout << "# of events: " << source.size() << std::endl;
    // std::cout << &source << std::endl;

  }

  HistoFiller(const HistoFiller& other) : histo(other.histo),
  source(other.source),
  filter(other.filter) {
    // std::cout << "COPY CTROR" << std::endl;
    // std::cout << "# of events: " << source.size() << std::endl;
  }
  
  
  void fill(unsigned int id) {
    // std::cout << "FILL histo: " << histo->GetName() << ": " << histo << std::endl;
    // std::cout << "   entry: " << id << " value: " << source[id] << std::endl;
    if(filter[id]) 
      histo->Fill(source[id]);
  }
  
  unsigned int nevents() const {
    // std::cout << "# of events: " << source.size() << std::endl;
    return source.size();
  }
  
  TH1F *histo;
  const std::vector<float> & source;
  const std::vector<bool> & filter;
  
};

class DataFrame {
public:
  // DataFrame() : branches() {}
  
  void addBranch(const std::string& name, std::vector<float> values) {
    branches[name] = values;
  }

  void addFilter(const std::string& name, std::vector<bool> values) {
    filters[name] = values;
  }

  bool knowsVariable(const std::string& name) const {
    if(branches.find(name) == branches.end()) return false;
    return true;
  }

  bool knowsSelection(const std::string& name) const {
    if(filters.find(name) == filters.end()) return false;
    return true;    
  }
  
  std::map<std::string, std::vector<float>> branches;
  std::map<std::string, std::vector<bool>> filters;

};



class FillerManager {
public:
  // FillerManager() {}
  
  void addVariable(const std::string& name, const std::vector<float>& values) {
    frame.addBranch(name, values);
  }

  void addFilter(const std::string& name, const std::vector<bool>& values) {
    frame.addFilter(name, values);
  }

  
  void add_1Dhisto(TH1F *histo, const std::string& source, const std::string& filter) {
    fillers.emplace_back(HistoFiller(histo, frame.branches[source], frame.filters[filter]));
  }
  
  bool knowsVariable(const std::string& name) const {
    return frame.knowsVariable(name);
  }
  
  bool knowsSelection(const std::string& name) const {
    return frame.knowsSelection(name);
  }
  
  void fill() {
    for(unsigned int id = 0; id < fillers[0].nevents(); id++) {
      for(auto & filler: fillers) {
        filler.fill(id);
      }
    }
    // for(auto & filler: fillers) {
    //   std::cout << filler.histo->GetName() << " has # entries: " << filler.histo->GetEntries() << std::endl;
    // }
  }
  
private:
  DataFrame frame;
  std::vector<HistoFiller> fillers;
};
