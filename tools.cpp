#include "tools.h"
#include <random> 
#include <algorithm>

using namespace std;

vector<int> getRandomIndices(int n) {
  vector<int> indicies(n);
  iota(indicies.begin(), indicies.end(), 0);
  random_device rd;
  mt19937 gen(rd());
  shuffle(indicies.begin(), indicies.end(), gen);
  return indicies;
}
