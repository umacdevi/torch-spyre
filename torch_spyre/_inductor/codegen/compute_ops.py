# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from torch_spyre._C import encode_constant, DataFormats
from torch_spyre._inductor.constants import DEPTHWISE_CONV2D_OP, CONV2D_DIM_LABELS
from sympy import Symbol
import traceback


def core_idx_to_slice_offset(
    arg,
    wk_slice: dict,
    work_slices: dict,
) -> int:
    offset = sum(arg.offsets.values())
    for dim, stride in arg.strides.items():
        if str(dim) in wk_slice and arg.scales[dim] > 0:
            offset += wk_slice[str(dim)] * stride // work_slices[dim]
    return offset


def num_bytes(df: DataFormats) -> int:
    """Try to avoid using this method; it is a bad API due to sub-byte datatypes"""
    num_elems = df.elems_per_stick()
    if num_elems > 128:
        raise RuntimeError(f"sub-byte dataformat {df}")
    return 128 // num_elems


def generate_constant_info(data_format, constants, num_cores):
    if len(constants.keys()) == 0:
        return "{}"
    constant_info = {}
    for name, value in constants.items():
        ci = {
            "dataFormat_": data_format.name,
            "name_": name,
            "data_": {
                "dim_prop_func": [{"Const": {}}, {"Const": {}}, {"Map": {}}],
                "dim_prop_attr": [
                    {"factor_": num_cores, "label_": "core"},
                    {"factor_": 1, "label_": "corelet"},
                    {"factor_": 1, "label_": "time"},
                ],
                "data_": {"[0, 0, 0]": [encode_constant(value, data_format)]},
            },
        }
        constant_info[f"{len(constant_info)}"] = ci
    return constant_info


def add_constant(kwargs, name, value) -> int:
    """
    Add a constant to kwargs['op_info']['constants'] and return its index.
    Returns:
        int: The index of the newly added constant (0-based)
    """
    # Ensure structure exists
    if "op_info" not in kwargs:
        kwargs["op_info"] = {}
    if "constants" not in kwargs["op_info"]:
        kwargs["op_info"]["constants"] = {}

    index = len(kwargs["op_info"]["constants"])
    kwargs["op_info"]["constants"][name] = value

    return index


def gen_coord_info_value(
    size: int,
    nsplits: int,
    elems_per_stick: int,
    is_stick_dim: bool,
    is_stick_reduction: bool = False,
    conv_params = {"conv_padding": "nopad", "total_size": -1}
):
    return (
        {
            "spatial": 3,
            "temporal": 0,
            "elemArr": 1,
            #"padding": "nopad",
            "padding": str(conv_params["conv_padding"]), 
            "folds": {
                "dim_prop_func": [
                    {
                        "Affine": {
                            "alpha_": size,
                            "beta_": 0,
                        }
                    },
                    {
                        "Affine": {
                            "alpha_": 0,
                            "beta_": 0,
                        }
                    },
                    {
                        "Affine": {
                            "alpha_": 0,
                            "beta_": 0,
                        }
                    },
                    {
                        "Affine": {
                            "alpha_": 1,
                            "beta_": 0,
                        }
                    },
                ],
                "dim_prop_attr": [
                    {
                        "factor_": nsplits,
                        "label_": "core_fold",
                    },
                    {
                        "factor_": 1,
                        "label_": "corelet_fold",
                    },
                    {
                        "factor_": 1,
                        "label_": "row_fold",
                    },
                    {
                        "factor_": conv_params["total_size"],
                        "label_": "elem_arr_0",
                    },
                ],
            },
        }
        if not is_stick_dim
        else {
            "spatial": 3,
            "temporal": 0,
            "elemArr": 2,
            "padding": "nopad",
            "folds": {
                "dim_prop_func": [
                    {
                        "Affine": {
                            "alpha_": elems_per_stick if is_stick_reduction else size,
                            "beta_": 0,
                        }
                    },
                    {
                        "Affine": {
                            "alpha_": 0,
                            "beta_": 0,
                        }
                    },
                    {
                        "Affine": {
                            "alpha_": 0,
                            "beta_": 0,
                        }
                    },
                    {
                        "Affine": {
                            "alpha_": elems_per_stick,
                            "beta_": 0,
                        }
                    },
                    {
                        "Affine": {
                            "alpha_": 0 if is_stick_reduction else 1,
                            "beta_": 0,
                        }
                    },
                ],
                "dim_prop_attr": [
                    {
                        "factor_": nsplits,
                        "label_": "core_fold",
                    },
                    {
                        "factor_": 1,
                        "label_": "corelet_fold",
                    },
                    {
                        "factor_": 1,
                        "label_": "row_fold",
                    },
                    {
                        "factor_": 1
                        if is_stick_reduction
                        else (size // elems_per_stick),
                        "label_": "elem_arr_1",
                    },
                    {
                        "factor_": elems_per_stick,
                        "label_": "elem_arr_0",
                    },
                ],
            },
        }
    )

def get_conv_params(tensor_num, dim, opfunc, conv_params, size):
    conv_padding = "nopad"
    total_size = size
    if tensor_num == 0 and opfunc == DEPTHWISE_CONV2D_OP:
        if ("pad_type" in conv_params and (str(dim) == str(conv_params["pad_dim_i"]) or str(dim) == str(conv_params["pad_dim_j"]))):
            conv_padding = conv_params["pad_type"] 
        if ("pad_dim_i" in conv_params and str(dim) == str(conv_params["pad_dim_i"]) and  "total_size_i" in conv_params):
            total_size = conv_params["total_size_i"]
        elif ("pad_dim_j" in conv_params and str(dim) == str(conv_params["pad_dim_j"]) and  "total_size_j" in conv_params):
            total_size = conv_params["total_size_j"]
    return {"conv_padding": conv_padding, "total_size": total_size}

    conv_padding=sdsc_spec.conv_params["pad_type"] if (i==0 and sdsc_spec.opfunc == DEPTHWISE_CONV2D_OP and "pad_type" in sdsc_spec.conv_params and (str(dim) == str(sdsc_spec.conv_params["pad_dim_i"]) or str(dim) == str(sdsc_spec.conv_params["pad_dim_j"]))) else "nopad"


def generate_sdsc(idx, sdsc_spec):
    print(f"In generate_sdsc: {sdsc_spec}")
    out_idx = len(sdsc_spec.args) - 1
    core_id_to_wk_slice = {
        str(c): {
            str(dim): int(expr.subs({Symbol("core_id"): c}))
            for dim, expr in sdsc_spec.core_id_to_work_slice.items()
        }
        for c in range(sdsc_spec.num_cores)
    }

    print(f"core_id_to_wk_slice: {core_id_to_wk_slice}")
    print(f"Contents for N_:")
    for dim, size in sdsc_spec.iteration_space.items():
        print(f"  {dim}: {size}")
    for label, layout_info in sdsc_spec.layouts.items():
        print(f"DIM ORDER: {layout_info['dim_order']}")
        print(f"STICK DIM ORDER: {layout_info['stick_dim_order'] if 'stick_dim_order' in layout_info else 'N/A'}")

    #return

    return {
        f"{idx}_{sdsc_spec.opfunc}": {
            "sdscFoldProps_": [{"factor_": 1, "label_": "time"}],
            "sdscFolds_": {
                "dim_prop_func": [{"Affine": {"alpha_": 1, "beta_": 0}}],
                "dim_prop_attr": [{"factor_": 1, "label_": "time"}],
                "data_": {"[0]": "0"},
            },
            "coreFoldProp_": {"factor_": sdsc_spec.num_cores, "label_": "core"},
            "coreletFoldProp_": {"factor_": 1, "label_": "corelet"},
            "numCoresUsed_": sdsc_spec.num_cores,
            "coreIdToDsc_": {str(c): 0 for c in range(sdsc_spec.num_cores)},
            "numWkSlicesPerDim_": {
                str(dim): num_wk_slices
                for dim, num_wk_slices in sdsc_spec.work_slices.items()
            },
            "coreIdToWkSlice_": core_id_to_wk_slice,
            "coreIdToDscSchedule": {
                str(c): [[-1, 0, 0, 0]] for c in range(sdsc_spec.num_cores)
            },
            "dscs_": [
                {
                    sdsc_spec.opfunc: {
                        "numCoresUsed_": sdsc_spec.num_cores,
                        "numCoreletsUsed_": 1,
                        "coreIdsUsed_": [c for c in range(sdsc_spec.num_cores)],
                        "N_": {
                            "name_": "n",
                            **{
                                str(dim) + "_": size
                                for dim, size in sdsc_spec.iteration_space.items()
                            },
                            **(
                                {
                                "paddingSizes_" :{
                                    #"i" : {"padFront_" : 1, "padBack_" : 1, "unneededPad_" : 0, "unneededPadFront_" : 0, "unneededPadBack_" : 0, "totalSize_" : 130, "stride_" : 1, "dilation_" : 1, "windowDim_" : "ki"},
                                    str(CONV2D_DIM_LABELS[2]) : {"padFront_":1, "padBack_":1, "totalSize_" : sdsc_spec.conv_params["total_size_i"], "stride_" : sdsc_spec.conv_params["stride_i"], "dilation_" : sdsc_spec.conv_params["dilation_i"], "windowDim_" : sdsc_spec.conv_params["window_dim_i"]},
                                    #str(CONV2D_DIM_LABELS[2]) : {"padFront_":0, "padBack_":0, "totalSize_" : sdsc_spec.conv_params["total_size_i"], "stride_" : sdsc_spec.conv_params["stride_i"], "dilation_" : sdsc_spec.conv_params["dilation_i"], "windowDim_" : sdsc_spec.conv_params["window_dim_i"]},
                                     #"j" : {"padFront_" : 1, "padBack_" : 1, "unneededPad_" : 0, "unneededPadFront_" : 0, "unneededPadBack_" : 0, "totalSize_" : 130, "stride_" : 1, "dilation_" : 1, "windowDim_" : "kj"}}
                                     str(CONV2D_DIM_LABELS[3]) : {"padFront_":1, "padBack_":1, "totalSize_" : sdsc_spec.conv_params["total_size_j"], "stride_" : sdsc_spec.conv_params["stride_j"], "dilation_" : sdsc_spec.conv_params["dilation_j"], "windowDim_" : sdsc_spec.conv_params["window_dim_j"]}
                                     #str(CONV2D_DIM_LABELS[3]) : {"padFront_":0, "padBack_":0, "totalSize_" : sdsc_spec.conv_params["total_size_j"], "stride_" : sdsc_spec.conv_params["stride_j"], "dilation_" : sdsc_spec.conv_params["dilation_j"], "windowDim_" : sdsc_spec.conv_params["window_dim_j"]}
                                    }
                                }
                                if sdsc_spec.opfunc == DEPTHWISE_CONV2D_OP
                                else {}
                             ),
                        },
                        #"coordinateMasking_": {
                        #    str(dim): mask_range
                        #    for dim, mask_range in sdsc_spec.coordinate_masking.items()
                        #},
                        "numCoreletsUsed_DSC2_": -1,
                        #"maskingConstId_": 0 if sdsc_spec.coordinate_masking else -1,
                        "dataStageParam_": {
                            "0": {
                                "ss_": {
                                    "name_": "core",
                                    **{
                                        str(dim) + "_": size
                                        // sdsc_spec.work_slices[dim]
                                        for dim, size in sdsc_spec.iteration_space.items()
                                    },
                                **(
                                    {
                                    "paddingSizes_" :{
                                        str(CONV2D_DIM_LABELS[2]) : {"padFront_":1, "padBack_":1, "totalSize_" : sdsc_spec.conv_params["total_size_i"], "stride_" : sdsc_spec.conv_params["stride_i"], "dilation_" : sdsc_spec.conv_params["dilation_i"], "windowDim_" : sdsc_spec.conv_params["window_dim_i"]},
                                        #str(CONV2D_DIM_LABELS[2]) : {"padFront_":0, "padBack_":0, "totalSize_" : sdsc_spec.conv_params["total_size_i"], "stride_" : sdsc_spec.conv_params["stride_i"], "dilation_" : sdsc_spec.conv_params["dilation_i"], "windowDim_" : sdsc_spec.conv_params["window_dim_i"]},
                                        str(CONV2D_DIM_LABELS[3]) : {"padFront_":1, "padBack_":1, "totalSize_" : sdsc_spec.conv_params["total_size_j"], "stride_" : sdsc_spec.conv_params["stride_j"], "dilation_" : sdsc_spec.conv_params["dilation_j"], "windowDim_" : sdsc_spec.conv_params["window_dim_j"]}
                                        #str(CONV2D_DIM_LABELS[3]) : {"padFront_":0, "padBack_":0, "totalSize_" : sdsc_spec.conv_params["total_size_j"], "stride_" : sdsc_spec.conv_params["stride_j"], "dilation_" : sdsc_spec.conv_params["dilation_j"], "windowDim_" : sdsc_spec.conv_params["window_dim_j"]}
                                        }
                                    }
                                    if sdsc_spec.opfunc == DEPTHWISE_CONV2D_OP
                                    else {}
                                 ),
                                },
                                "el_": {
                                    "name_": "core",
                                    **{
                                        str(dim) + "_": size
                                        // sdsc_spec.work_slices[dim]
                                        for dim, size in sdsc_spec.iteration_space.items()
                                    },
                                **(
                                    {
                                    "paddingSizes_" :{
                                        str(CONV2D_DIM_LABELS[2]) : {"padFront_":1, "padBack_":1, "totalSize_" : sdsc_spec.conv_params["total_size_i"], "stride_" : sdsc_spec.conv_params["stride_i"], "dilation_" : sdsc_spec.conv_params["dilation_i"], "windowDim_" : sdsc_spec.conv_params["window_dim_i"]},
                                        #str(CONV2D_DIM_LABELS[2]) : {"padFront_":0, "padBack_":0, "totalSize_" : sdsc_spec.conv_params["total_size_i"], "stride_" : sdsc_spec.conv_params["stride_i"], "dilation_" : sdsc_spec.conv_params["dilation_i"], "windowDim_" : sdsc_spec.conv_params["window_dim_i"]},
                                        str(CONV2D_DIM_LABELS[3]) : {"padFront_":1, "padBack_":1, "totalSize_" : sdsc_spec.conv_params["total_size_j"], "stride_" : sdsc_spec.conv_params["stride_j"], "dilation_" : sdsc_spec.conv_params["dilation_j"], "windowDim_" : sdsc_spec.conv_params["window_dim_j"]}
                                        #str(CONV2D_DIM_LABELS[3]) : {"padFront_":0, "padBack_":0, "totalSize_" : sdsc_spec.conv_params["total_size_j"], "stride_" : sdsc_spec.conv_params["stride_j"], "dilation_" : sdsc_spec.conv_params["dilation_j"], "windowDim_" : sdsc_spec.conv_params["window_dim_j"]}
                                        }
                                    }
                                    if sdsc_spec.opfunc == DEPTHWISE_CONV2D_OP
                                    else {}
                                 ),
                                },
                            },
                        },
                        "primaryDsInfo_": {
                            label: {
                                "layoutDimOrder_": [
                                    str(dim) for dim in layout_info["dim_order"]
                                ],
                                "stickDimOrder_": [str(layout_info["stick_dim_order"])],
                                "stickSize_": [layout_info["stick_size"]],
                            }
                            for label, layout_info in sdsc_spec.layouts.items()
                        },
                        "scheduleTree_": [
                            {
                                "nodeType_": "allocate",
                                "name_": f"allocate-Tensor{i}_{'lx' if 'lx' in tensor.allocation else 'hbm'}",
                                "prev_": "",
                                "ldsIdx_": i,
                                "component_": "hbm" if not tensor.allocation else "lx",
                                **(
                                    {
                                    "padding_" :{
                                         str(CONV2D_DIM_LABELS[2]) : sdsc_spec.conv_params["pad_type"],
                                         str(CONV2D_DIM_LABELS[3]) : sdsc_spec.conv_params["pad_type"]
                                        }
                                    }
                                    if sdsc_spec.opfunc == DEPTHWISE_CONV2D_OP and i == 0
                                    else {}
                                 ),
                                "layoutDimOrder_": [
                                    str(dim)
                                    for dim in sdsc_spec.layouts[tensor.layout][
                                        "dim_order"
                                    ]
                                ],
                                #"layoutDimOrder_": ([
                                #    str(dim)
                                #    for dim in sdsc_spec.layouts[tensor.layout][
                                #        "dim_order"
                                #    ]]
                                #    if i != 2 else  ["mb", "j", "i", "in"]
                                #),
                                "maxDimSizes_": [
                                    tensor.max_dim_sizes[dim]
                                    for dim in sdsc_spec.layouts[tensor.layout][
                                        "dim_order"
                                    ]
                                ],
                                "startAddressCoreCorelet_": {
                                    "dim_prop_func": [
                                        {"Map": {}},
                                        {"Const": {}},
                                        {"Const": {}},
                                    ],
                                    "dim_prop_attr": [
                                        {
                                            "factor_": sdsc_spec.num_cores,
                                            "label_": "core",
                                        },
                                        {"factor_": 1, "label_": "corelet"},
                                        {"factor_": 1, "label_": "time"},
                                    ],
                                    "data_": {
                                        f"[{c}, 0, 0]": str(
                                            tensor.start_address
                                            + core_idx_to_slice_offset(
                                                tensor,
                                                core_id_to_wk_slice[str(c)],
                                                sdsc_spec.work_slices,
                                            )
                                            * num_bytes(tensor.data_format)
                                        )
                                        if "lx" not in tensor.allocation
                                        else str(tensor.start_address)
                                        for c in range(sdsc_spec.num_cores)
                                        #  lx addr is baked into tensor.start_addr already
                                    },
                                },
                                #**(
                                #    {
                                #        "backGapCore_": {
                                #            str(dim): {
                                #                "-1": str(gap)  # HBM is -1
                                #            }
                                #            for dim, gap in tensor.backGap.items()
                                #        }
                                #    }
                                #    if tensor.backGap
                                #    else {}
                                #),
                                "coordinates_": {
                                    "coordInfo": {
                                        str(dim): gen_coord_info_value(
                                            size=sdsc_spec.iteration_space[dim]
                                            // sdsc_spec.work_slices[dim]
                                            if (tensor.scales[dim] == 1) and dim in sdsc_spec.iteration_space
                                            else 1,
                                            nsplits=sdsc_spec.work_slices[dim]
                                            if (tensor.scales[dim] == 1 and dim in sdsc_spec.iteration_space)
                                            else 1,
                                            elems_per_stick=tensor.data_format.elems_per_stick(),
                                            is_stick_dim=(
                                                sdsc_spec.layouts[tensor.layout][
                                                    "stick_dim_order"
                                                ].has(dim)
                                            ),
                                            is_stick_reduction=(
                                                tensor.scales[dim] == -2
                                            ),
                                            #conv_padding=sdsc_spec.conv_params["pad_type"] if (i==0 and sdsc_spec.opfunc == DEPTHWISE_CONV2D_OP and "pad_type" in sdsc_spec.conv_params and (str(dim) == str(sdsc_spec.conv_params["pad_dim_i"]) or str(dim) == str(sdsc_spec.conv_params["pad_dim_j"]))) else "nopad"
                                            conv_params=get_conv_params(i, dim, sdsc_spec.opfunc, sdsc_spec.conv_params, sdsc_spec.iteration_space[dim])
                                        )
                                        for dim in sdsc_spec.layouts[tensor.layout][
                                            "dim_order"
                                        ]
                                    },
                                    "coreIdToWkSlice_": {},
                                },
                            }
                            for i, tensor in enumerate(sdsc_spec.args)
                        ],
                        "pdsRelation_": { "isPdsReuse": 1},
                        "labeledDs_": [
                            {
                                "ldsIdx_": i,
                                "dsName_": f"Tensor{i}",
                                "dsType_": tensor.layout,
                                "scale_": [
                                    tensor.scales[dim]
                                    for dim in sdsc_spec.layouts[tensor.layout][
                                        "dim_order"
                                    ]
                                ],
                                "wordLength": num_bytes(tensor.data_format),
                                "dataFormat_": tensor.data_format.name,
                                "memOrg_": (
                                    {
                                        "hbm": {"isPresent": 1, "isPadded": 1, "isZeroPadded": 0},
                                        "lx": {"isPresent": 1, "isPadded": 1, "isZeroPadded": 1},
                                    }
                                    if (i==0 and sdsc_spec.opfunc == DEPTHWISE_CONV2D_OP) 
                                    else {
                                        "hbm": {"isPresent": 1},
                                        "lx": {"isPresent": 1},
                                    }
                                )
                                if not tensor.allocation
                                else (
                                  {
                                    "lx": {
                                      "isPresent": 1, "isPadded": 1
                                    }
                                  }
                                  if (i==0 and sdsc_spec.opfunc == DEPTHWISE_CONV2D_OP)
                                  else {"lx" : {"isPresent": 1}}
                                ),
                            }
                            for i, tensor in enumerate(sdsc_spec.args)
                        ],
                        "constantInfo_": generate_constant_info(
                            sdsc_spec.data_format,
                            sdsc_spec.constants,
                            sdsc_spec.num_cores,
                        ),
                        "computeOp_": [
                            {
                                "exUnit": sdsc_spec.execution_unit,
                                "opFuncName": sdsc_spec.opfunc,
                                "attributes_": {
                                    "dataFormat_": sdsc_spec.data_format.name,
                                    "fidelity_": "regular",
                                },
                                "location": "Inner",
                                "inputLabeledDs": [
                                    f"Tensor{i}-idx{i}"
                                    for i in range(sdsc_spec.num_inputs)
                                ],
                                "outputLabeledDs": [f"Tensor{out_idx}-idx{out_idx}"],
                            }
                        ],
                    }
                }
            ],
        }
    }
