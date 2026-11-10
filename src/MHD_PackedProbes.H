#pragma once

#include "MHD_ProbeTrajectory.H"
#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <memory>
#include <hdf5.h>
#ifdef PR_MPI
#include <mpi.h>
#endif

namespace HelioCubed
{
inline bool packedProbeFile(std::string filename)
{
    for(char& c : filename) c=std::tolower(static_cast<unsigned char>(c));
    const auto dot=filename.find_last_of('.');
    return dot!=std::string::npos &&
        (filename.substr(dot)==".h5" || filename.substr(dot)==".hdf5");
}

inline bool sameProbeFile(const std::string& a,const std::string& b)
{
    std::error_code error;
    if(std::filesystem::equivalent(a,b,error)) return true;
    return std::filesystem::weakly_canonical(a)==std::filesystem::weakly_canonical(b);
}

// Check the complete configuration before opening any output, including aliases
// through symlinks/hardlinks and convergence-study output suffixes.
inline void validateProbeFiles(const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs,const std::string& suffix="")
{
    for(size_t i=0;i<inputs.size();++i)
    {
        if(inputs[i].empty() || outputs[i].empty())
            throw std::invalid_argument("Each probe needs coords and filename");
        for(size_t j=0;j<inputs.size();++j)
        {
            if(sameProbeFile(outputs[i]+suffix,inputs[j]))
                throw std::invalid_argument("Probe output must differ from every trajectory input");
            if(i!=j && (packedProbeFile(inputs[i]) || packedProbeFile(inputs[j]))
                && sameProbeFile(outputs[i]+suffix,outputs[j]+suffix))
                throw std::invalid_argument("Each packed probe set requires a distinct output file");
        }
    }
}

namespace ProbeH5
{
inline hid_t check(hid_t status)
{
    if(status<0) throw std::runtime_error("Packed probe HDF5 operation failed");
    return status;
}

// All handles are scoped, including error paths. No HDF5 or MPI work is deferred
// to static destruction (which happens after MPI_Finalize in the driver).
struct Handle
{
    hid_t id;
    herr_t (*close)(hid_t);
    Handle(hid_t value,herr_t (*closer)(hid_t)) : id(check(value)),close(closer) {}
    ~Handle() {close(id);}
    Handle(const Handle&)=delete;
    Handle& operator=(const Handle&)=delete;
    operator hid_t() const {return id;}
};
inline std::vector<hsize_t> shape(hid_t dataset)
{
    Handle space(H5Dget_space(dataset),H5Sclose);
    const int rank=check(H5Sget_simple_extent_ndims(space));
    std::vector<hsize_t> dims(rank);
    check(H5Sget_simple_extent_dims(space,dims.data(),nullptr));
    return dims;
}
inline void attribute(hid_t file,const char* name,const std::string& value)
{
    Handle type(H5Tcopy(H5T_C_S1),H5Tclose);
    check(H5Tset_size(type,value.size()+1));
    Handle space(H5Screate(H5S_SCALAR),H5Sclose);
    Handle attr(H5Acreate2(file,name,type,space,H5P_DEFAULT,H5P_DEFAULT),H5Aclose);
    check(H5Awrite(attr,type,value.c_str()));
}
inline void integerAttribute(hid_t file,const char* name,int value)
{
    Handle space(H5Screate(H5S_SCALAR),H5Sclose);
    Handle attr(H5Acreate2(file,name,H5T_NATIVE_INT,space,H5P_DEFAULT,H5P_DEFAULT),H5Aclose);
    check(H5Awrite(attr,H5T_NATIVE_INT,&value));
}
inline int integerAttribute(hid_t file,const char* name)
{
    Handle attr(H5Aopen(file,name,H5P_DEFAULT),H5Aclose);
    Handle space(H5Aget_space(attr),H5Sclose);
    if(H5Sget_simple_extent_type(space)!=H5S_SCALAR)
        throw std::runtime_error("Invalid packed probe scalar attribute");
    int value;
    check(H5Aread(attr,H5T_NATIVE_INT,&value));
    return value;
}
inline std::string attribute(hid_t file,const char* name)
{
    Handle attr(H5Aopen(file,name,H5P_DEFAULT),H5Aclose);
    Handle type(H5Aget_type(attr),H5Tclose);
    Handle space(H5Aget_space(attr),H5Sclose);
    if(H5Sget_simple_extent_type(space)!=H5S_SCALAR)
        throw std::runtime_error("Invalid packed probe scalar attribute");
    if(H5Tget_class(type)!=H5T_STRING || H5Tis_variable_str(type)>0)
        throw std::runtime_error("Invalid packed probe string attribute");
    std::vector<char> value(H5Tget_size(type)+1,0);
    check(H5Aread(attr,type,value.data()));
    return value.data();
}
inline void strings(hid_t file,const char* name,const std::vector<std::string>& values)
{
    size_t width=1;
    for(const auto& value : values) width=std::max(width,value.size()+1);
    std::vector<char> data(width*values.size(),0);
    for(size_t i=0;i<values.size();++i)
        std::copy(values[i].begin(),values[i].end(),data.begin()+i*width);
    Handle type(H5Tcopy(H5T_C_S1),H5Tclose);
    check(H5Tset_size(type,width));
    const hsize_t count=values.size();
    Handle space(H5Screate_simple(1,&count,nullptr),H5Sclose);
    Handle dataset(H5Dcreate2(file,name,type,space,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT),H5Dclose);
    check(H5Dwrite(dataset,type,H5S_ALL,H5S_ALL,H5P_DEFAULT,data.data()));
}
inline void create(hid_t file,const char* name,hid_t type,std::vector<hsize_t> dims)
{
    auto max=dims,chunks=dims;
    max[0]=H5S_UNLIMITED;
    chunks[0]=dims.size()==1 ? 256 : 1;
    if(dims.size()>1) chunks[1]=std::min<hsize_t>(dims[1],256);
    Handle space(H5Screate_simple(dims.size(),dims.data(),max.data()),H5Sclose);
    Handle properties(H5Pcreate(H5P_DATASET_CREATE),H5Pclose);
    check(H5Pset_chunk(properties,dims.size(),chunks.data()));
    check(H5Pset_shuffle(properties));
    check(H5Pset_deflate(properties,4));
    Handle dataset(H5Dcreate2(file,name,type,space,H5P_DEFAULT,properties,H5P_DEFAULT),H5Dclose);
}
inline void append(hid_t file,const char* name,hid_t type,
    std::vector<hsize_t> dims,hsize_t step,const void* data)
{
    Handle dataset(H5Dopen2(file,name,H5P_DEFAULT),H5Dclose);
    dims[0]=step+1;
    check(H5Dset_extent(dataset,dims.data()));
    Handle space(H5Dget_space(dataset),H5Sclose);
    std::vector<hsize_t> start(dims.size(),0);
    start[0]=step;dims[0]=1;
    check(H5Sselect_hyperslab(space,H5S_SELECT_SET,start.data(),nullptr,dims.data(),nullptr));
    Handle memory(H5Screate_simple(dims.size(),dims.data(),nullptr),H5Sclose);
    check(H5Dwrite(dataset,type,memory,space,H5P_DEFAULT,data));
}
}

// Output row order matches the text path after decimal year:
// r, latitude, simulation longitude, n, VR, VT, VN, P, BR, BT, BN.
using ProbeRow=std::array<double,11>;

class PackedProbeSet
{
    std::string m_input,m_fingerprint;
    std::vector<ProbeTrajectory> m_trajectories;
    std::vector<double> m_positions,m_values;
    std::vector<int> m_found;

    void write(const std::string& output,double epoch)
    {
        using namespace ProbeH5;
        if(sameProbeFile(m_input,output))
            throw std::invalid_argument("Packed probe input and output must differ");
        const hsize_t n=size();
        for(size_t p=0;p<size();++p) if(m_found[p])
        {
            for(size_t c=0;c<8;++c) if(!std::isfinite(m_values[p*8+c]))
                throw std::runtime_error("Nonfinite packed probe sample");
        }
        const bool exists=std::filesystem::exists(output);
        Handle file(exists ? H5Fopen(output.c_str(),H5F_ACC_RDWR,H5P_DEFAULT) :
            H5Fcreate(output.c_str(),H5F_ACC_EXCL,H5P_DEFAULT,H5P_DEFAULT),H5Fclose);
        hsize_t step=0;
        if(exists)
        {
            if(attribute(file,"format_name")!="HelioCubed packed probe output" ||
                integerAttribute(file,"format_version")!=1 ||
                attribute(file,"trajectory_fingerprint")!=m_fingerprint ||
                attribute(file,"trajectory_source")!=m_input)
                throw std::runtime_error("Existing packed probe output is incompatible with these trajectories");
            Handle time(H5Dopen2(file,"/time",H5P_DEFAULT),H5Dclose);
            const auto dims=shape(time);
            if(dims.size()!=1) throw std::runtime_error("Invalid packed probe time shape");
            step=dims[0];
            for(const auto& spec : std::vector<std::pair<std::string,std::vector<hsize_t>>>{
                {"/position",{step,n,3}},{"/values",{step,n,8}},{"/found",{step,n}}})
            {
                Handle dataset(H5Dopen2(file,spec.first.c_str(),H5P_DEFAULT),H5Dclose);
                if(shape(dataset)!=spec.second)
                    throw std::runtime_error("Existing packed probe output has incompatible dataset dimensions");
            }
        }
        else
        {
            attribute(file,"format_name","HelioCubed packed probe output");
            integerAttribute(file,"format_version",1);
            integerAttribute(file,"probe_count",int(n));
            integerAttribute(file,"variable_count",8);
            attribute(file,"trajectory_source",m_input);
            attribute(file,"trajectory_fingerprint",m_fingerprint);
            attribute(file,"probe_order","radius, latitude_offset, longitude_offset (longitude fastest)");
            attribute(file,"time_units","decimal year");
            attribute(file,"vector_basis","RTN");
            strings(file,"/field_name",{"number_density","VR","VT","VN","P","BR","BT","BN"});
            strings(file,"/field_units",{"cm^-3","km/s","km/s","km/s","picodyne/cm^2","microgauss","microgauss","microgauss"});
            // Follow MS-FLUKSS's packed position order, which differs from text rows.
            strings(file,"/position_field_name",{"distance_au","longitude_deg","latitude_deg"});
            strings(file,"/position_units",{"AU","degree","degree"});
            attribute(file,"longitude_convention","simulation longitude = HGI + 181.02 degrees, wrapped to [0,360)");
            Handle input(H5Fopen(m_input.c_str(),H5F_ACC_RDONLY,H5P_DEFAULT),H5Fclose);
            for(const char* name : {"/axes","/probe_name"})
                if(check(H5Lexists(input,name,H5P_DEFAULT))>0)
                    check(H5Ocopy(input,name,file,name,H5P_DEFAULT,H5P_DEFAULT));
            create(file,"/time",H5T_IEEE_F64LE,{0});
            create(file,"/position",H5T_IEEE_F64LE,{0,n,3});
            create(file,"/values",H5T_IEEE_F64LE,{0,n,8});
            create(file,"/found",H5T_STD_U8LE,{0,n});
        }
        std::vector<unsigned char> found(size());
        for(size_t p=0;p<size();++p)
        {
            if(m_found[p]>1) throw std::runtime_error("Packed probe has multiple MPI owners");
            found[p]=m_found[p]!=0;
            if(!found[p])
            {
                std::fill_n(m_positions.begin()+p*3,3,std::numeric_limits<double>::quiet_NaN());
                std::fill_n(m_values.begin()+p*8,8,std::numeric_limits<double>::quiet_NaN());
            }
        }
        append(file,"/position",H5T_NATIVE_DOUBLE,{0,n,3},step,m_positions.data());
        append(file,"/values",H5T_NATIVE_DOUBLE,{0,n,8},step,m_values.data());
        append(file,"/found",H5T_NATIVE_UCHAR,{0,n},step,found.data());
        // Time is written last; inconsistent extents after an interrupted write
        // are rejected when reopening, rather than silently mixing records.
        append(file,"/time",H5T_NATIVE_DOUBLE,{0},step,&epoch);
        check(H5Fflush(file,H5F_SCOPE_GLOBAL));
    }
public:
    explicit PackedProbeSet(const std::string& input)
        : m_input(std::filesystem::canonical(input).string())
    {
        using namespace ProbeH5;
        Handle file(H5Fopen(m_input.c_str(),H5F_ACC_RDONLY,H5P_DEFAULT),H5Fclose);
        Handle dataset(H5Dopen2(file,"/trajectories",H5P_DEFAULT),H5Dclose);
        const auto dims=shape(dataset);
        if(dims.size()!=5 || dims[4]!=5 || dims[3]==0)
            throw std::runtime_error("/trajectories must have shape [radius, latitude, longitude, time, 5]");
        size_t n=1;
        for(int d=0;d<3;++d)
        {
            if(!dims[d] || dims[d]>size_t(std::numeric_limits<int>::max()/8)/n)
                throw std::runtime_error("Packed probe count exceeds the MPI count limit");
            n*=dims[d];
        }
        if(dims[3]>std::numeric_limits<size_t>::max()/5)
            throw std::runtime_error("Packed trajectory time axis is too large");
        m_trajectories.resize(n);
        // Read angular tiles in the input's chunk order. Keep temporary memory
        // bounded to one latitude/longitude tile over time.
        std::uint64_t fingerprint=14695981039346656037ULL;
        auto hash=[&](const void* value,size_t bytes)
        {
            const auto* data=static_cast<const unsigned char*>(value);
            for(size_t i=0;i<bytes;++i) {fingerprint^=data[i];fingerprint*=1099511628211ULL;}
        };
        for(auto dim : dims) hash(&dim,sizeof(dim));
        Handle space(H5Dget_space(dataset),H5Sclose);
        for(hsize_t r=0;r<dims[0];++r)
        for(hsize_t lat=0;lat<dims[1];lat+=8)
        for(hsize_t lon=0;lon<dims[2];lon+=8)
        {
            hsize_t start[]={r,lat,lon,0,0};
            hsize_t count[]={1,std::min<hsize_t>(8,dims[1]-lat),std::min<hsize_t>(8,dims[2]-lon),dims[3],5};
            size_t total=1;
            for(auto dim : count)
            {
                if(dim>std::numeric_limits<size_t>::max()/sizeof(double)/total)
                    throw std::runtime_error("Packed trajectory tile is too large");
                total*=dim;
            }
            std::vector<double> raw(total);
            check(H5Sselect_hyperslab(space,H5S_SELECT_SET,start,nullptr,count,nullptr));
            Handle memory(H5Screate_simple(5,count,nullptr),H5Sclose);
            check(H5Dread(dataset,H5T_NATIVE_DOUBLE,memory,space,H5P_DEFAULT,raw.data()));
            hash(raw.data(),raw.size()*sizeof(double));
            for(hsize_t a=0;a<count[1];++a)
            for(hsize_t b=0;b<count[2];++b)
            {
                auto& trajectory=m_trajectories[(r*dims[1]+lat+a)*dims[2]+lon+b];
                for(hsize_t t=0;t<dims[3];++t)
                {
                    const double* row=raw.data()+((a*count[2]+b)*dims[3]+t)*5;
                    for(int c=0;c<5;++c) if(!std::isfinite(row[c]))
                        throw std::runtime_error("Packed trajectory contains nonfinite coordinates");
                    if(row[0]<1 || row[0]>9999 || row[0]!=std::floor(row[0]))
                        throw std::runtime_error("Packed trajectory year must be an integer in [1,9999]");
                    const int year=int(row[0]),days=gregorianLeapYear(year) ? 366 : 365;
                    if(row[1]<1 || row[1]>=days+1 || row[2]<=0)
                        throw std::runtime_error("Invalid packed trajectory day or radius");
                    const double epoch=year+(row[1]-1)/days;
                    if(!trajectory.time.empty() && epoch<=trajectory.time.back())
                        throw std::runtime_error("Packed trajectory times must strictly increase");
                    trajectory.time.push_back(epoch);trajectory.radius.push_back(row[2]);
                    trajectory.latitude.push_back(row[3]);trajectory.longitude.push_back(row[4]);
                }
            }
        }
        std::ostringstream fingerprintText;fingerprintText<<std::hex<<fingerprint;
        m_fingerprint=fingerprintText.str();
        m_positions.resize(n*3);m_values.resize(n*8);m_found.resize(n);
    }
    size_t size() const {return m_trajectories.size();}
    const ProbeTrajectory& trajectory(size_t p) const {return m_trajectories.at(p);}
    void begin()
    {
        std::fill(m_positions.begin(),m_positions.end(),0);
        std::fill(m_values.begin(),m_values.end(),0);
        std::fill(m_found.begin(),m_found.end(),0);
    }
    void record(size_t p,const ProbeRow& row)
    {
        m_found.at(p)=1;
        m_positions[p*3]=row[0];m_positions[p*3+1]=row[2];m_positions[p*3+2]=row[1];
        std::copy(row.begin()+3,row.end(),m_values.begin()+p*8);
    }
    void finish(const std::string& output,double epoch)
    {
        int rank=0;
#ifdef PR_MPI
        MPI_Comm_rank(MPI_COMM_WORLD,&rank);
        // One reduction per array for the whole set, rather than per probe.
        auto reduce=[&](auto& data,MPI_Datatype type)
        {
            MPI_Reduce(rank==0 ? MPI_IN_PLACE : data.data(),data.data(),
                int(data.size()),type,MPI_SUM,0,MPI_COMM_WORLD);
        };
        reduce(m_positions,MPI_DOUBLE);reduce(m_values,MPI_DOUBLE);reduce(m_found,MPI_INT);
#endif
        int error=0;
        std::string message;
        if(rank==0)
        {
            try {write(output,epoch);}
            catch(const std::exception& e) {error=1;message=e.what();}
        }
#ifdef PR_MPI
        MPI_Bcast(&error,1,MPI_INT,0,MPI_COMM_WORLD);
#endif
        if(error) throw std::runtime_error("Could not write packed probes to "+output+
            (message.empty() ? " (see rank 0 error)" : ": "+message));
    }
};

inline PackedProbeSet& packedProbeSet(const std::string& input)
{
    static std::map<std::string,std::unique_ptr<PackedProbeSet>> cache;
    auto& set=cache[input];
    if(!set) set=std::make_unique<PackedProbeSet>(input);
    return *set;
}
}
