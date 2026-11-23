"""Анимация полного поля выбранного квадратичного слоя 175 м, 0–1 с."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
from typing import cast

import numpy as np
from PIL import Image
from scripts.project_paths import PROJECT_ROOT
os.environ.setdefault("MPLCONFIGDIR",str(PROJECT_ROOT/".matplotlib"))
import matplotlib
matplotlib.use("Agg")
from matplotlib.animation import FFMpegWriter, writers
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.image import AxesImage

from scripts.dev_1_4_server_preparation.profile_golden_analysis import (
    ALPHA0, DATA, IMAGES, load_profile, number, plt, read_json,
)
from scripts.dev_1_4_server_preparation.compare_smoke import file_evidence
from scripts.dev_1_4_server_preparation.velocity_animate import (
    FULL_EXTENT, Fields, FloatArray, Movie, Pixels, decorate, mapped, pixel_map, sha256,
)
from scripts.reporting import mapping, sequence

ROOT = DATA / "profile_golden_fields_1s_3060_v1"
OUTPUT = IMAGES / "profile_golden_fields_1s_v1"
REPORT = DATA / "profile_golden_fields_1s_analysis_v1/animations.json"
WINNER = DATA / "profile_golden_1s_3060_v1/runs/eval_09"
FACTOR = .5684038628158912
FPS = 10
PREVIEW_FRAME = 50
REPEAT_RELATIVE_TOLERANCE = 1e-10
REPEAT_ABSOLUTE_TOLERANCE = 1e-18


def record_file(path: Path) -> dict[str,object]:
    return {"path":str(path.resolve()),"sha256":sha256(path),"bytes":path.stat().st_size}


def repeat_agreement(values: FloatArray, reference: FloatArray, times: FloatArray) -> dict[str,object]:
    if values.shape!=reference.shape or values.shape[0]!=len(times) or not np.isfinite(values).all() or not np.isfinite(reference).all():
        raise ValueError("Invalid surface repeat arrays")
    denominator=float(np.linalg.norm(reference))
    if denominator<=0:raise ValueError("Zero surface reference")
    delta=values-reference
    relative=float(np.linalg.norm(delta)/denominator);maximum=float(np.max(np.abs(delta)))
    if relative>REPEAT_RELATIVE_TOLERANCE or maximum>REPEAT_ABSOLUTE_TOLERANCE:
        raise ValueError("Surface repeat exceeds established global tolerances")
    late=(times>=.5)&(times<=1.)
    late_norm=float(np.linalg.norm(reference[late]))
    return {"exact_values":bool(np.array_equal(values,reference)),"numerically_equivalent":True,
        "relative_l2":relative,"max_abs_m_s":maximum,
        "thresholds":{"relative_l2_max":REPEAT_RELATIVE_TOLERANCE,"absolute_max_m_s":REPEAT_ABSOLUTE_TOLERANCE},
        "late_window_independent":{"time_window_s":[.5,1.],"relative_l2":float(np.linalg.norm(delta[late])/late_norm),
            "relative_delta_objective":float((np.linalg.norm(values[late])-late_norm)/late_norm)},
        "meaning":"Global repeat comparison passes the pre-existing thresholds; not bitwise identity or an independent physical-accuracy check"}


def load_fields(directory: Path) -> tuple[Fields,dict[str,object]]:
    metadata=read_json(directory/"metadata.json")
    validation=read_json(directory/"validation.json")
    winner=load_profile(FACTOR,WINNER,9)
    if metadata.get("status")!="completed" or validation.get("status")!="passed":
        raise ValueError("Need successfully completed all-node field repeat")
    if mapping(validation["metadata"])["sha256"]!=sha256(directory/"metadata.json"):
        raise ValueError("Run validation belongs to different metadata")
    failed_path=directory/"metadata.failed.json"
    if failed_path.exists() and validation.get("approved_failed_metadata_sha256")!=sha256(failed_path):
        raise ValueError("Failed record lacks explicit SHA-bound postvalidation")
    if metadata.get("field_components")!=["Vx","Vy"] or metadata.get("field_cadence_s")!=.01 or metadata.get("field_shape")!=[101,589541,2]:
        raise ValueError("Unexpected field component/cadence/shape metadata")
    if mapping(metadata["recorded_counts"]).get("fields")!=101 or mapping(metadata["recorded_counts"]).get("surface")!=1001:
        raise ValueError("Need101 field frames and1001 surface samples")
    for key in ("source_input_sha256","source_node_id","initial_state","physical_model","dt_s","steps_count","nodes_sha256","abc_edge_count","nodal_alpha_sha256","run_spec"):
        if metadata[key]!=winner.run.metadata[key]:
            raise ValueError(f"Field repeat differs from selected candidate: {key}")
    for key in ("native","mesh_evidence","geometry_evidence"):
        if mapping(metadata[key])["sha256"]!=mapping(winner.run.metadata[key])["sha256"]:
            raise ValueError(f"Field repeat evidence differs: {key}")
    profile=mapping(metadata["profile"])
    if profile["shape"]!="power" or profile["exponent"]!=2. or profile["amplitude_factor"]!=FACTOR:
        raise ValueError("Wrong damping profile")
    runtime=mapping(metadata["runtime"])
    if runtime["result_code"]!=runtime["time_end_code"] or runtime["steps_count"]!=96000 or not np.isclose(number(runtime["current_time_s"]),1.,rtol=0,atol=2e-10):
        raise ValueError("Incomplete numerical run")
    evidence:dict[str,object]={}
    for name in ("nodes.npy","field_time_s.npy","velocity_m_s.npy"):
        actual=record_file(directory/name)
        expected=mapping(mapping(metadata["field_files"])[name])
        if actual["sha256"]!=expected["sha256"] or actual["bytes"]!=expected["bytes"]:
            raise ValueError(f"Field SHA/size mismatch: {name}")
        evidence[name]=actual
    coordinates=cast(FloatArray,np.load(directory/"nodes.npy",mmap_mode="r",allow_pickle=False))
    times=cast(FloatArray,np.load(directory/"field_time_s.npy",allow_pickle=False))
    velocity=cast(FloatArray,np.load(directory/"velocity_m_s.npy",mmap_mode="r",allow_pickle=False))
    if coordinates.shape!=(589541,2) or times.shape!=(101,) or velocity.shape!=(101,589541,2):
        raise ValueError("Unexpected full-field shape")
    if any(array.dtype!=np.float64 for array in (coordinates,times,velocity)):
        raise ValueError("Fields must preserve float64")
    if not np.isfinite(coordinates).all() or not np.array_equal(times,np.linspace(0.,1.,101)):
        raise ValueError("Invalid coordinates or field time axis")
    if hashlib.sha256(coordinates.tobytes()).hexdigest()!=metadata["nodes_sha256"] or not np.array_equal(coordinates[winner.run.surface.node_ids],winner.run.surface.coordinates):
        raise ValueError("Global node order or receiver-node coordinates differ")
    if not np.allclose(coordinates.min(axis=0),[-175.,-875.],rtol=0,atol=1e-9) or not np.allclose(coordinates.max(axis=0),[875.,0.],rtol=0,atol=1e-9):
        raise ValueError("Expected the complete175m padded domain")
    if np.any(velocity[0]!=0):
        raise ValueError("Initial field must be zero")
    surface_path=directory/"surface.npz"
    if sha256(surface_path)!=mapping(metadata["surface_file"])["sha256"]:
        raise ValueError("Surface SHA mismatch")
    with np.load(surface_path,allow_pickle=False) as archive:
        if set(archive.files)!={"time_s","coordinates_m","node_ids","velocity_m_s"}:
            raise ValueError("Unexpected surface schema")
        if not np.array_equal(archive["time_s"],winner.run.surface.times) or not np.array_equal(archive["coordinates_m"],winner.run.surface.coordinates) or not np.array_equal(archive["node_ids"],winner.run.surface.node_ids):
            raise ValueError("Surface observation axes changed")
        agreement=repeat_agreement(np.asarray(archive["velocity_m_s"]),winner.run.surface.velocity,winner.run.surface.times)
        if not np.array_equal(velocity[:,winner.run.surface.node_ids],archive["velocity_m_s"][::10]):
            raise ValueError("All-node and surface samples disagree")
    recorded=mapping(validation["surface_match"])
    if recorded!=metadata["surface_match"] or recorded.get("numerically_equivalent") is not True or recorded.get("exact_values")!=agreement["exact_values"] or recorded.get("thresholds")!=agreement["thresholds"]:
        raise ValueError("Repeat validation does not agree with independently checked arrays")
    for key in ("relative_l2","max_abs_m_s"):
        if not np.isclose(number(recorded[key]),number(agreement[key]),rtol=1e-9,atol=0):
            raise ValueError("Recorded repeat metric disagrees with arrays")
    threshold_source=Path(__file__).with_name("verify_velocity_repeat.py")
    if mapping(recorded["threshold_source_file"])["sha256"]!=sha256(threshold_source) or validation.get("all_node_surface_values_exact") is not True:
        raise ValueError("Missing threshold provenance or exact intra-run field/surface check")
    agreement.update(recorded_comparison=recorded,threshold_source_file=record_file(threshold_source),
        preserved_failed_metadata=record_file(failed_path) if failed_path.exists() else None)
    return Fields(directory,metadata,coordinates,times,velocity,evidence),agreement


def scan(fields: Fields) -> dict[str,object]:
    speed_max=component_max=0.
    per_frame:list[dict[str,float]]=[]
    for index,time in enumerate(fields.times):
        values=np.asarray(fields.velocity[index])
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite velocity at field frame{index}")
        speed=np.hypot(values[:,0],values[:,1])
        maximum=float(np.max(speed));component=float(np.max(np.abs(values)))
        speed_max=max(speed_max,maximum);component_max=max(component_max,component)
        per_frame.append({"time_s":float(time),"speed_max_m_s":maximum,"component_abs_max_m_s":component,
            "rms_vector_m_s":float(np.sqrt(np.mean(speed**2)))})
    if speed_max<=0 or component_max<=0:
        raise ValueError("Cannot animate an entirely zero run")
    return {"speed_max_m_s":speed_max,"component_abs_max_m_s":component_max,
        "per_frame":per_frame,"meaning":"Unweighted nodal velocity samples; no physical-energy or exact-error interpretation"}


def make_movie(pixels: Pixels, diagnostics: dict[str,object]) -> Movie:
    speed_max=number(diagnostics["speed_max_m_s"])*1e9
    component_max=number(diagnostics["component_abs_max_m_s"])*1e9
    speed_norm=LogNorm(vmin=speed_max*1e-6,vmax=speed_max)
    signed_norm=SymLogNorm(linthresh=component_max*1e-5,vmin=-component_max,vmax=component_max,base=10)
    speed_cmap=plt.get_cmap("inferno").copy();speed_cmap.set_bad("black");speed_cmap.set_under("black")
    figure,axes=plt.subplots(1,3,figsize=(18,7.2),layout="constrained")
    artists:list[AxesImage]=[]
    for index,(axis,title) in enumerate(zip(axes,("Модуль скорости |v|","Компонента Vx","Компонента Vy"),strict=True)):
        artist=axis.imshow(np.zeros(pixels.shape),extent=pixels.extent,origin="lower",interpolation="nearest",
            norm=speed_norm if index==0 else signed_norm,cmap=speed_cmap if index==0 else "RdBu_r")
        decorate(axis,title,FULL_EXTENT,True)
        axis.set_xticks([-175,0,350,700,875]);axis.set_yticks([-875,-700,-350,0])
        artists.append(artist)
    figure.colorbar(artists[0],ax=axes[0],label="|v|, нм/с · log",shrink=.8,pad=.025)
    figure.colorbar(artists[1],ax=list(axes[1:]),label=f"Vx/Vy, нм/с · SymLog; линейно до {component_max*1e-5:.2g}",shrink=.8,pad=.025)
    title=f"Однородная среда · слой 175 м · k={FACTOR:.6f} · αmax={FACTOR*ALPHA0:.3f} с⁻¹\nVp=4087 м/с · Vs=2431,62 м/с · p4 · источник Рикера 30 Гц, x=350 м"
    label=figure.suptitle(title+" · t=0,00 с",fontsize=13)
    figure.supxlabel("Пунктир — основная область 700×700 м; x вправо, y вверх · шкалы неизменны весь ролик\n|v|: нуль и значения ниже порога — чёрные; Vx/Vy: нуль белый · шаг физического времени 0,01 с; воспроизведение 10 кадров/с",fontsize=10)
    return Movie("velocity_fields",figure,artists,label,title)


def update(movie: Movie, fields: Fields, pixels: Pixels, index: int) -> None:
    values=np.asarray(fields.velocity[index])
    arrays=(np.hypot(values[:,0],values[:,1]),values[:,0],values[:,1])
    for artist,values_ in zip(movie.artists,arrays,strict=True):
        artist.set_data(mapped(values_,pixels)*1e9)
    movie.time_label.set_text(f"{movie.title} · t={fields.times[index]:.2f} с")


def video_evidence(path: Path, frames: int) -> dict[str,object]:
    process=subprocess.run(["ffprobe","-v","error","-count_frames","-select_streams","v:0","-show_entries",
        "stream=width,height,nb_read_frames,r_frame_rate,duration,pix_fmt","-of","json",str(path)],check=True,capture_output=True,text=True)
    stream=mapping(sequence(mapping(json.loads(process.stdout))["streams"])[0])
    if stream.get("nb_read_frames")!=str(frames) or stream.get("r_frame_rate")!="10/1" or stream.get("pix_fmt")!="yuv420p" or not np.isclose(float(str(stream["duration"])),frames/FPS,rtol=0,atol=1e-5):
        raise ValueError("Encoded video frame count/rate/duration differs")
    return {**record_file(path),"ffprobe":stream}


def make_gif(video: Path, output: Path, frames: int) -> dict[str,object]:
    graph="fps=10,scale=1080:-2:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle"
    subprocess.run(["ffmpeg","-v","error","-n","-threads","2","-i",str(video),"-filter_complex_threads","2",
        "-filter_complex",graph,"-loop","0",str(output)],check=True)
    durations:list[int]=[]
    with Image.open(output) as gif:
        count=getattr(gif,"n_frames",1)
        for index in range(count):
            gif.seek(index);durations.append(int(gif.info.get("duration",0)))
        size=list(gif.size)
    if count!=frames or durations!=[100]*frames:
        raise ValueError("GIF does not retain every frame and10fps timing")
    if output.stat().st_size>15_000_000:
        raise ValueError("GIF exceeded the15MB inline-preview budget")
    return {**record_file(output),"frame_count":count,"duration_ms":sum(durations),"size_px":size,
        "derived_from":"MP4 preview at1080px width,128-color shared palette; full-resolution video is primary"}


def render(fields: Fields, output: Path, pixels: Pixels, diagnostics: dict[str,object], *, check_mode: bool=False) -> tuple[dict[str,object],dict[str,object],list[dict[str,object]]]:
    movie=make_movie(pixels,diagnostics)
    writer=FFMpegWriter(fps=FPS,codec="libx264",extra_args=["-pix_fmt","yuv420p","-crf","19","-threads","2","-movflags","+faststart"])
    previews:list[dict[str,object]]=[]
    video=output/"velocity_fields.mp4"
    try:
        with writer.saving(movie.figure,str(video),dpi=100):
            for index in range(len(fields.times)):
                update(movie,fields,pixels,index);writer.grab_frame()
                if index==(1 if check_mode else PREVIEW_FRAME):
                    path=output/"velocity_fields_t050.png"
                    movie.figure.savefig(path,dpi=130)
                    previews.append({**record_file(path),"physical_time_s":float(fields.times[index])})
                if index%20==0:
                    print(json.dumps({"stage":"rendered","frame":index+1,"frames":len(fields.times)}),flush=True)
    finally:
        plt.close(movie.figure)
    return video_evidence(video,len(fields.times)),make_gif(video,output/"velocity_fields_preview.gif",len(fields.times)),previews


def animate(directory: Path, output: Path, report: Path) -> dict[str,object]:
    directory.resolve().relative_to(DATA);output.resolve().relative_to(IMAGES);report.resolve().relative_to(DATA)
    if output.exists() or report.exists() or report.parent.exists():
        raise FileExistsError("Choose fresh animation and report directories")
    if not writers.is_available("ffmpeg"):
        raise RuntimeError("ffmpeg is required")
    fields,agreement=load_fields(directory);diagnostics=scan(fields)
    pixels=pixel_map(fields.coordinates,FULL_EXTENT,1050/630)
    output.mkdir(parents=True,exist_ok=False)
    video,gif,previews=render(fields,output,pixels,diagnostics)
    speed_max=number(diagnostics["speed_max_m_s"]);component_max=number(diagnostics["component_abs_max_m_s"])
    result:dict[str,object]={"status":"completed","schema":"chapter14.profile_golden_fields_1s_animations.v1",
        "utc":datetime.now(timezone.utc).isoformat(),"selected_factor":FACTOR,"alpha_peak_per_direction_s_inv":FACTOR*ALPHA0,
        "frame_count":101,"physical_end_s":1.,"physical_cadence_s":.01,"playback_fps":FPS,"playback_duration_s":10.1,
        "playback_slowdown_between_frames":10.,"timing_note":"First-to-last physical interval1s is shown over10s; the last frame is held0.1s, total encoded duration10.1s",
        "components":["Vx","Vy"],"panels":["hypot(Vx,Vy)","Vx","Vy"],
        "color_scales":{"display_units":"nm/s; source arrays remainm/s","speed_log_limits_m_s":[speed_max*1e-6,speed_max],
            "component_symlog_limits_m_s":[-component_max,component_max],"component_linear_threshold_m_s":component_max*1e-5,
            "fixed_all_frames":True,"policy":"One physical scale per quantity across all101 frames; no per-frame amplification. Speed zero/below floor black; signed zero white"},
        "rendering":{"method":"Nearest SEM node at regular pixel centers; origin lower; no temporal interpolation",
            "physical_extent_m":list(FULL_EXTENT),"pixel_shape":list(pixels.shape),"pixel_spacing_m":1050/630,
            "maximum_pixel_to_node_distance_m":pixels.maximum_distance_m,"core_box_m":[0.,700.,-700.,0.]},
        "field_diagnostics":diagnostics,"inputs":{"directory":str(directory.resolve()),"metadata":record_file(directory/"metadata.json"),
            "validation":record_file(directory/"validation.json"),"fields":fields.files,"selected_surface":record_file(WINNER/"surface.npz"),
            "surface_repeat_exact":agreement["exact_values"],"numerical_repeat_verified":True,
            "surface_repeat_comparison":agreement,"all_node_surface_samples_exact":True},
        "videos":[video],"gif":gif,"previews":previews,"code":file_evidence(Path(__file__)),
        "shared_helpers":file_evidence(Path(__file__).with_name("velocity_animate.py")),
        "interpretation":"One selected homogeneous175m experiment; velocity visualization with nonlinear fixed color scales, not energy or an independent validation of absorbing boundaries"}
    report.parent.mkdir(parents=True,exist_ok=False)
    with report.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")
    return result


def cpu_check(output: Path) -> dict[str,object]:
    output.resolve().relative_to(DATA)
    if output.exists():raise FileExistsError("Choose new CPU-check directory")
    with tempfile.TemporaryDirectory(prefix="tmp_profile_golden_animate_",dir=PROJECT_ROOT) as temporary:
        directory=Path(temporary)
        xs=np.linspace(-175.,875.,31);ys=np.linspace(-875.,0.,26)
        xx,yy=np.meshgrid(xs,ys);coords=np.column_stack((xx.ravel(),yy.ravel()))
        values=np.zeros((3,len(coords),2))
        values[1,:,0]=(coords[:,0]-350.)*1e-11;values[1,:,1]=(coords[:,1]+400.)*1e-11
        values[2]=-values[1]*.1
        fields=Fields(directory,{},coords,np.array([0.,.5,1.]),values,{})
        pixels=pixel_map(coords,FULL_EXTENT,35.)
        diagnostics=scan(fields);movie=make_movie(pixels,diagnostics)
        update(movie,fields,pixels,1)
        for index,expected in enumerate((np.hypot(values[1,:,0],values[1,:,1]),values[1,:,0],values[1,:,1])):
            np.testing.assert_array_equal(np.asarray(movie.artists[index].get_array()),mapped(expected,pixels)*1e9)
        limits=[(artist.norm.vmin,artist.norm.vmax) for artist in movie.artists]
        update(movie,fields,pixels,2)
        if limits!=[(artist.norm.vmin,artist.norm.vmax) for artist in movie.artists]:raise AssertionError("Per-frame color rescaling")
        plt.close(movie.figure)
        times=np.linspace(0.,1.,1001);reference=np.ones((1001,70,2),dtype=np.float64)*1e-8
        repeated=reference.copy();repeated[550,0,0]+=1e-20
        agreement=repeat_agreement(repeated,reference,times)
        if agreement["exact_values"] is not False or agreement["numerically_equivalent"] is not True:
            raise AssertionError("Tolerance comparison did not distinguish numerical and bitwise identity")
        repeated[550,0,0]+=1e-12
        try:repeat_agreement(repeated,reference,times)
        except ValueError:pass
        else:raise AssertionError("Out-of-tolerance repeat was accepted")
        video,gif,_=render(fields,directory,pixels,diagnostics,check_mode=True)
        if mapping(video["ffprobe"])["nb_read_frames"]!="3" or gif["frame_count"]!=3:
            raise AssertionError("Incomplete synthetic video/GIF")
    result:dict[str,object]={"status":"passed","utc":datetime.now(timezone.utc).isoformat(),"scope":"CPU rendering/encoding checks; no wave solver",
        "checks":["Mapped component signs and vector magnitude agree exactly with source arrays","Color scales remain fixed across frames",
            "Numerical-repeat comparison distinguishes bitwise difference within established tolerances and rejects a larger discrepancy",
            "MP4 andGIF retain three synthetic frames at10fps; codec/frame count/duration verified","Temporary previews removed"],
        "code":file_evidence(Path(__file__))}
    output.mkdir(parents=True,exist_ok=False)
    with (output/"validation.json").open("x",encoding="utf-8") as stream:json.dump(result,stream,ensure_ascii=False,indent=2);stream.write("\n")
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("animate","check"))
    parser.add_argument("--input",type=Path,default=ROOT/"run")
    parser.add_argument("--out",type=Path,default=OUTPUT)
    parser.add_argument("--report",type=Path,default=REPORT)
    parser.add_argument("--checks",type=Path,default=DATA/"profile_golden_fields_1s_animation_cpu_v1")
    args=parser.parse_args()
    result=cpu_check(args.checks) if args.command=="check" else animate(args.input,args.out,args.report)
    print(json.dumps({"status":result["status"],"frames":result.get("frame_count")}),flush=True)


if __name__=="__main__":main()
