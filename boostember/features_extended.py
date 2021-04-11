import os,time,sys


features = {'histogram':256,
 'byteentropy':256,
 'strings':104,
 'general':10,
 'header':62,
 'section':255,
 'imports':1280,
 'exports':128,
 'datadirectories':30}


def datadir_name():
    dataarr = []
    datadirs = ["EXPORT_TABLE", "IMPORT_TABLE", "RESOURCE_TABLE", "EXCEPTION_TABLE", "CERTIFICATE_TABLE",
                "BASE_RELOCATION_TABLE", "DEBUG", "ARCHITECTURE", "GLOBAL_PTR", "TLS_TABLE", "LOAD_CONFIG_TABLE",
                "BOUND_IMPORT", "IAT", "DELAY_IMPORT_DESCRIPTOR", "CLR_RUNTIME_HEADER"]
    for i,val in enumerate(datadirs):
            if i < len(datadirs):
                dataarr.append(f"datadirectories_{val}_size")
                dataarr.append(f"datadirectories_{val}_virtual_address")
    return dataarr


def columngen_name(dim=256, colname='ember'):
    return [f'{colname}_{i}' for i in range(0,dim)]


def emberfeaturesheader():
    colname = []
    #histogram 256
    colname.extend(columngen_name(256, 'histogram_'))
    #byteentropy 256
    colname.extend(columngen_name(256, 'byteentropy_'))
    #strings 104
    colname.extend(['strings_numstrings', 'strings_avlength', 'strings_printables'])
    colname.extend(columngen_name(96, 'strings_printabledist'))
    colname.extend(['strings_entropy', 'strings_paths', 'strings_urls', 'strings_registry', 'strings_MZ'])
    #general 10
    colname.extend(['general_size', 'general_vsize', 'general_has_debug', 'general_exports', 'general_imports', 'general_has_relocations', 'general_has_resources', 'general_has_signature', 'general_has_tls', 'general_symbols'])
    #header 62
    colname.extend(['header_coff_timestamp'])
    colname.extend(columngen_name(10, 'header_coff_machine'))
    colname.extend(columngen_name(10, 'header_coff_characteristics'))
    colname.extend(columngen_name(10, 'header_optional_subsystem'))
    colname.extend(columngen_name(10, 'header_optional_dll_characteristics'))
    colname.extend(columngen_name(10, 'header_optional_magic'))
    colname.extend(['header_optional_major_image_version', 'header_optional_minor_image_version', 'header_optional_major_linker_version', 'header_optional_minor_linker_version', 'header_optional_major_operating_system_version', 'header_optional_minor_operating_system_version', 'header_optional_major_subsystem_version', 'header_optional_minor_subsystem_version', 'header_optional_sizeof_code', 'header_optional_sizeof_headers', 'header_optional_sizeof_heap_commit'])
    #section 255
    colname.extend(['section_size', 'section_size_nonzero', 'section_size_empty', 'section_size_RX', 'section_size_W'])
    colname.extend(columngen_name(50, 'section_sizes_hashed'))
    colname.extend(columngen_name(50, 'section_entropy_hashed'))
    colname.extend(columngen_name(50, 'section_vsize_hashed'))
    colname.extend(columngen_name(50, 'section_entry_name_hashed'))
    colname.extend(columngen_name(50, 'section_characteristics_hashed'))
    #imports 1280
    colname.extend(columngen_name(256, 'imports_libraries_hashed'))
    colname.extend(columngen_name(1024, 'imports_hashed'))
    #exports 128
    colname.extend(columngen_name(128, 'exports_hashed'))
    #datadirectories 30
    colname.extend(datadir_name())
    return colname