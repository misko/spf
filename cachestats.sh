#!/bin/bash

# cachestats for dm-cache
#
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2023-2024 Forza <forza@tnonline.net>


# Help text
usage() {
	echo "Usage: $(basename "$0") [-v|--verbose] [DEVICE_NAME or PATH] [DEVICE_NAME or PATH] ..."
	echo "Options:"
	echo "  -h, --help      Display this help message"
	echo "  -v, --verbose   Display detailed information"
}
# Check for help option or no arguments
if [ "$#" -eq 0 ]; then
    usage
    exit 1
fi
if [[ "${1:0:1}" == "-" ]]; then
	case "$1" in
		-h|--help)
			usage
			exit 0
			;;
		-v|--verbose)
			VERBOSE=1
			shift
			;;
		*)
			echo "Invalid option: $1"
			usage
			exit 1
			;;
	esac
fi

# Declare variables
declare -A data
declare -a dmstatus
declare -a expanded_devices
declare -i sizess
declare status_output
declare device_name_or_path
declare device_name
declare separator_needed

# Function to convert bytes to IEC units
to_iec() {
	local -i bytes kib mib gib tib # integers
	bytes=$1
	kib=$(( (bytes ) / 1024 ))
	mib=$((   (kib ) / 1024 ))
	gib=$((   (mib ) / 1024 ))
	tib=$((   (gib ) / 1024 ))
	if [ $tib -gt 0 ]; then
		echo "$tib TiB"
	elif [ $gib -gt 0 ]; then
		echo "$gib GiB"
	elif [ $mib -gt 0 ]; then
		echo "$mib MiB"
	elif [ $kib -gt 0 ]; then
		echo "$kib KiB"
	else
		echo "$bytes bytes"
	fi
}
debug_output(){
	# DEBUG info

	# Field order of `dmsetup status <cachedev>`:
	# https://www.kernel.org/doc/html/latest/admin-guide/device-mapper/cache.html
	#
	# <metadata block size> <#used metadata blocks>/<#total metadata blocks>
	# <cache block size> <#used cache blocks>/<#total cache blocks>
	# <#read hits> <#read misses> <#write hits> <#write misses>
	# <#demotions> <#promotions> <#dirty> <#features> <features>*
	# <#core args> <core args>* <policy name> <#policy args> <policy args>*
	# <cache metadata mode>

	# Loop through through all fields to list and print their values
	if (( DEBUG > 0 )) ; then
		printf "\nDEBUG INFO\n"
		printf "========\n"
		printf "dmstatus index: value\n"
		## Ouput `dmsetup status` raw data:
		for ((i = 0; i < ${#dmstatus[@]}; i += 1)); do
			echo "# $i: ${dmstatus[$i]}"
		done
		printf "\nidentified values:\n"
			printf "========\n"
		for field in "${!data[@]}"; do
				value="${data[$field]}"
				if [ -n "$value" ]; then
					echo "$field : $value"
				else
					echo "Field '$field' not found."
				fi
			done | sort
		printf "========\n"
	fi
}

# Expand wildcard patterns for devices
for arg in "$@"; do
	if [[ "$arg" == /* ]]; then
		# Full path provided, use as-is
		expanded_devices+=("$arg")
	else
		# Assume partial name; expand in /dev/mapper
		# shellcheck disable=SC2231
		for device in /dev/mapper/${arg}*; do
			if [[ -e $device ]]; then
				expanded_devices+=("$device")
			else
				echo "No matching devices for pattern: $arg" >&2
			fi
		done
	fi
done

# Add separator if there is more than one device
separator_needed=$(( ${#expanded_devices[@]} > 1 ))

# Loop through all devices and print their info
for device_name_or_path in "${expanded_devices[@]}"; do

	# Strip path part from device name
	device_name=$(basename "${device_name_or_path}")

	# Get the status output using dmsetup
	status_output=$(dmsetup status "${device_name}" 2>/dev/null)

	# Check if the device exists and has valid output
	if [ -n "$status_output" ]; then
		# Check device block size
		sizess=$(blockdev --getss "/dev/mapper/${device_name}")

		# Parse the dmstatus information into the associative array
		IFS='/: ' read -r -a dmstatus <<< "${status_output}"

		# Populate the data array
		data["name"]="$device_name"
		data["origin_start"]=$(( dmstatus[0] * sizess )) # in sectors
		data["origin_length"]=$(( dmstatus[1] * sizess )) # in sectors
		data["table_type"]="${dmstatus[2]}"
		data["metadata_block_size"]=$(( dmstatus[3] * sizess )) # in sectors
		data["used_metadata_blocks"]="${dmstatus[4]}"
		data["total_metadata_blocks"]="${dmstatus[5]}"
		data["cache_block_size"]=$(( dmstatus[6] * sizess )) # in sectors
		data["used_cache_blocks"]="${dmstatus[7]}"
		data["total_cache_blocks"]="${dmstatus[8]}"
		data["read_hits"]="${dmstatus[9]}"
		data["read_misses"]="${dmstatus[10]}"
		data["write_hits"]="${dmstatus[11]}"
		data["write_misses"]="${dmstatus[12]}"
		data["demotions"]="${dmstatus[13]}"
		data["promotions"]="${dmstatus[14]}"
		data["dirty_cache"]=$(( dmstatus[15] * data[cache_block_size] ))
		if [ "${dmstatus[16]}" = 2 ]; then
			data["cache_type"]="${dmstatus[17]}"
			data["discard_passdown"]="${dmstatus[18]}"
			data["migration_threshold"]=$(( dmstatus[21] * sizess ))
			data["cache_policy"]="${dmstatus[22]}"
			data["smq_count"]="${dmstatus[23]}"
			data["cache_rw"]="${dmstatus[24]}"
			data["status"]="${dmstatus[25]//-/OK}"
		else
			data["cache_type"]="${dmstatus[17]}"
			data["discard_passdown"]="true"
			data["migration_threshold"]=$(( dmstatus[20] * sizess ))
			data["cache_policy"]="${dmstatus[21]}"
			data["smq_count"]="${dmstatus[22]}"
			data["cache_rw"]="${dmstatus[23]}"
			data["status"]="${dmstatus[24]//-/OK}"
		fi

				# Print dm-cache data
		if (( separator_needed )); then
			printf "\n\n********************** DEVICE %s **********************\n" "${data[name]}"
		fi

		# Print debug information
		debug_output
		
		printf "\nDEVICE\n========\n"
		printf "%-*s%s\n" "26" "Device-mapper name: " "/dev/mapper/${data[name]}"
		printf "%-*s%s\n" "26" "Origin size: " "$(to_iec $(( data[origin_length] - data[origin_start] )) )"

		if [[ $VERBOSE -eq 1 ]]; then
			printf "%-*s%s\n" "26" "Discards: " "${data[discard_passdown]}"
		fi

		printf "\n"
		printf "CACHE\n========\n"
		printf "%-*s%s\n" "26" "$(( data[cache_block_size] ))"
		printf "%-*s%s\n" "26" "Size / Usage: " "$(to_iec $(( data[total_cache_blocks] * data[cache_block_size] ))) / $(to_iec $(( data[used_cache_blocks] * data[cache_block_size] ))) ($(( 100 * data[used_cache_blocks] / data[total_cache_blocks] )) %)"
		printf "%-*s%s\n" "26" "Read Hit Rate: " "${data[read_hits]} / $(( data[read_misses] + data[read_hits] )) ($(( 100 * data[read_hits] / (data[read_hits] + data[read_misses] ) )) %)"
		printf "%-*s%s\n" "26" "Write Hit Rate: " "${data[write_hits]} / $(( data[write_misses] + data[write_hits] )) ($(( 100 * data[write_hits] / ( data[write_hits] + data[write_misses] ) )) %)"
		printf "%-*s%s\n" "26" "Dirty: " "$(to_iec "${data[dirty_cache]}")"

		if [[ $VERBOSE -eq 1 ]]; then
			printf "%-*s%s\n" "26" "Block Size: " "$(to_iec "${data[cache_block_size]}")"
			printf "%-*s%s\n" "26" "Promotions / Demotions: " "${data[promotions]} / ${data[demotions]}"
			printf "%-*s%s\n" "26" "Migration Threshold: " "$(to_iec "${data[migration_threshold]}")"
			printf "%-*s%s\n" "26" "Read-Write mode: " "${data[cache_rw]}"
			printf "%-*s%s\n" "26" "Type: " "${data[cache_type]}"
			printf "%-*s%s\n" "26" "Policy: " "${data[cache_policy]}"
			printf "%-*s%s\n" "26" "Status: " "${data[status]}"
	
			printf "\n"
			printf "METADATA\n"
			printf "========\n"
			printf "%-*s%s\n" "26" "Size / Usage: " "$(to_iec $(( data[total_metadata_blocks] * data[metadata_block_size] ))) / $(to_iec $(( data[used_metadata_blocks] * data[metadata_block_size] ))) ($(( 100 * data[used_metadata_blocks] / data[total_metadata_blocks] )) %)"
		fi
	else
		echo "Device ${device_name_or_path} not found or no valid status output."
	fi
done
