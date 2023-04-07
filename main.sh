while getopts i:a:o: flag
do
    case "${flag}" in
        i) image=${OPTARG};;
        a) audio=${OPTARG};;
        o) output=${OPTARG};;
    esac
done
# echo $image $audio $output
filename=$(basename -- "$audio")
extension_audio="${filename##*.}"
filename_audio="${filename%.*}"
image=$(realpath "$image")
audio=$(realpath "$audio")

cd src/face_generator
python demo.py --id Obama2 --driving_audio $audio

cd ../../src/face_reenactment
python demo.py --config config/vox-256-spade.yaml --checkpoint ./config/00000189-checkpoint.pth.tar --source_image $image --driving_video ../../src/face_generator/results/Obama2/$filename_audio/$filename_audio.avi --result_video ./output.mp4 --relative --adapt_scale --find_best_frame

cd ../../src/face_res
python demo.py ../../src/face_reenactment/ad-output.mp4 sr save sound

cd ../../
mkdir $output
cp src/face_reenactment/ad-output_result_srTrue_sound.mp4 $output/output.mp4
rm -rf src/face_generator/results
rm -rf src/face_reenactment/ad-output*
rm -rf src/face_reenactment/output*