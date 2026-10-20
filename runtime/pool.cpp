#include "pool.h"
#include <algorithm>
#include <cstring>
#include <set>
#include <sstream>
#include <stdexcept>

namespace ge_runtime {
RequestPool LoadPool(const IoSpec &spec, const nlohmann::json &inputs) {
    std::vector<char> bytes;
    if(!ReadBinFile(inputs.at("file").get<std::string>(),bytes)) throw std::runtime_error("cannot read packed inputs");
    RequestPool pool;std::set<std::string> shapes,names;
    pool.maxBytes.assign(spec.inputs.size(),0);
    pool.maxOutputBytes.assign(spec.outputs.size(),0);
    for(const auto &item:inputs.at("cases")) {
        Request request;request.name=item.at("id").get<std::string>();request.samples=item.at("samples").get<int64_t>();
        if(request.name.empty() || !names.insert(request.name).second || request.samples<1) throw std::runtime_error("invalid case ID/sample count");
        const auto &tensors=item.at("tensors");
        if(tensors.size()!=spec.inputs.size()) throw std::runtime_error("input count differs from graph");
        std::set<std::string> tensorNames;
        for(const auto &tensor:tensors) if(!tensorNames.insert(tensor.at("name").get<std::string>()).second) throw std::runtime_error("duplicate input name");
        std::ostringstream key;
        for(size_t index=0;index<spec.inputs.size();++index) {
            const auto &node=spec.inputs[index];const nlohmann::json *found=nullptr;
            for(const auto &tensor:tensors) if(tensor.at("name")==node.logical) found=&tensor;
            if(!found) throw std::runtime_error("missing named input "+node.logical);
            TensorPlan plan;plan.logical=node.logical;plan.node=node.node.empty()?node.logical:node.node;
            plan.dtype=found->at("dtype").get<std::string>();plan.format=node.format;
            plan.shape=found->at("shape").get<std::vector<int64_t>>();
            if(plan.dtype!=node.dtype || plan.shape.size()!=node.shape.size()) throw std::runtime_error("input dtype/rank mismatch");
            for(size_t axis=0;axis<node.shape.size();++axis) if(node.shape[axis]>=0 && node.shape[axis]!=plan.shape[axis]) throw std::runtime_error("static dimension mismatch");
            size_t size=plan.Bytes(),offset=found->at("offset").get<size_t>();
            if(found->at("bytes").get<size_t>()!=size || offset>bytes.size() || size>bytes.size()-offset) throw std::runtime_error("packed input byte range mismatch");
            request.data.emplace_back(bytes.begin()+offset,bytes.begin()+offset+size);
            request.plans.push_back(plan);key<<ShapeToString(plan.shape)<<';';
            pool.maxBytes[index]=std::max(pool.maxBytes[index],size);
        }
        for(size_t index=0;index<spec.outputs.size();++index) {
            size_t capacity=item.at("output_capacities").at(spec.outputs[index].logical).get<size_t>();
            if(capacity==0) throw std::runtime_error("output capacity must be positive");
            pool.maxOutputBytes[index]=std::max({pool.maxOutputBytes[index],capacity,spec.outputs[index].max_bytes});
        }
        request.shapeKey=key.str();shapes.insert(request.shapeKey);pool.requests.push_back(std::move(request));
    }
    if(pool.Empty()) throw std::runtime_error("empty input pool");
    pool.distinctShapes=shapes.size();return pool;
}
} // namespace ge_runtime
