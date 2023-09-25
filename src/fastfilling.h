#include <iostream>
#include <vector>
// unsigned int fillTH1Fast1(const std::vector<float> & corrArr) {
// std::cout << "SIZE: " << corrArr.size() << std::endl;
// return corrArr.size();
// }
// 
// #include <boost/histogram.hpp>

class HistoFiller {
public:
  
  virtual void fill(unsigned int id) = 0;  

  virtual unsigned int nevents() const = 0;
  
};



class HistoFiller1D : public HistoFiller {
public:
  HistoFiller1D(TH1F *histo, const std::vector<float> & source, const std::vector<bool> & filter) :
  histo(histo),
  filter(filter),
  source(source) {
    // std::cout << "Add histo: " << histo->GetName() << ": " << histo << std::endl;
    // std::cout << "# of events: " << source.size() << std::endl;
    // std::cout << &source << std::endl;

  }

  HistoFiller1D(const HistoFiller1D& other) : histo(other.histo),
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



class HistoFiller2D : public HistoFiller {
public:
  HistoFiller2D(TH2F *histo, const std::vector<float> & sourceX, const std::vector<float> & sourceY, const std::vector<bool> & filter) :
  histo(histo),
  filter(filter),
  sourceX(sourceX),
  sourceY(sourceY) {
    // std::cout << "Add histo: " << histo->GetName() << ": " << histo << std::endl;
    // std::cout << "# of events: " << sourceX.size() << std::endl;
    // std::cout << &sourceX << std::endl;
  // std::cout << &sourceY << std::endl;

  }

  HistoFiller2D(const HistoFiller2D& other) : histo(other.histo),
  sourceX(other.sourceX),
  sourceY(other.sourceY),
  filter(other.filter) {
    // std::cout << "COPY CTROR" << std::endl;
    // std::cout << "# of events: " << source.size() << std::endl;
  }
  
  
  void fill(unsigned int id) {
    // std::cout << "FILL histo: " << histo->GetName() << ": " << histo << std::endl;
    // std::cout << "   entry: " << id << " value X: " << sourceX[id] << " Y: " << sourceY[id] << std::endl;
    if(filter[id]) 
      histo->Fill(sourceX[id], sourceY[id]);
  }
  
  unsigned int nevents() const {
    // std::cout << "# of events: " << source.size() << std::endl;
    return sourceX.size();
  }
  
  TH2F *histo;
  const std::vector<float> & sourceX;
  const std::vector<float> & sourceY;
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
    fillers.push_back(new HistoFiller1D(histo, frame.branches[source], frame.filters[filter]));
  }

  void add_2Dhisto(TH2F *histo, const std::string& sourceX, const std::string& sourceY, const std::string& filter) {
    fillers.push_back(new HistoFiller2D(histo, frame.branches[sourceX], frame.branches[sourceY], frame.filters[filter]));
  }
  
  bool knowsVariable(const std::string& name) const {
    return frame.knowsVariable(name);
  }
  
  bool knowsSelection(const std::string& name) const {
    return frame.knowsSelection(name);
  }
  
  void fill() {
    for(unsigned int id = 0; id < fillers[0]->nevents(); id++) {
      for(auto filler: fillers) {
        filler->fill(id);
      }
    }
    // for(auto & filler: fillers) {
    //   std::cout << filler.histo->GetName() << " has # entries: " << filler.histo->GetEntries() << std::endl;
    // }
  }
  
private:
  DataFrame frame;
  std::vector<HistoFiller *> fillers;
};


void fill1D_rate(TH1* histo, const std::vector<float>& values) {
  for (auto value: values) {
    for(int bin = 1; bin <= histo->FindBin(value); bin++) {
      histo->Fill(histo->GetBinCenter(bin));
    }
  }
}

// def effForRate(rate):
//      cut = 9999
//      for ix in xrange(1,rateplot.GetNbinsX()+1):
//          if rateplot.GetBinContent(ix) <= rate:
//              cut = rateplot.GetXaxis().GetBinLowEdge(ix)
//              break


void fill_1Dhist(TH1* hist, const std::vector<float>& array, const std::vector<float>& weights) {
  // FIXME: check sizes

  for (unsigned i =0; i < array.size(); ++i)
    hist->Fill(array[i], weights[i]);
}

void fill_1Dhist(TH1* hist, const std::vector<float>& array) {

  for (auto value: array)
    hist->Fill(value);
}

void fill_2Dhist(TH2* hist, const std::vector<float>& arrayX, const std::vector<float>& arrayY, const std::vector<float>& weights) {
    // FIXME: check sizes

  for(unsigned i =0; i < arrayX.size(); ++i) {
    hist->Fill(arrayX[i], arrayY[i], weights[i]);
  }
}

void fill_2Dhist(TH2* hist, const std::vector<float>& arrayX, const std::vector<float>& arrayY) {
  // FIXME: check sizes
  for(unsigned i =0; i < arrayX.size(); ++i) {
    hist->Fill(arrayX[i], arrayY[i]);
  }
}
