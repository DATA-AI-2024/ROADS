import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import logo from "./logo.svg";
import "./App.css";
import NaverMap from "./NaverMap";
import { socket } from "./socket";
import { log } from "console";
import { unstable_batchedUpdates } from "react-dom";
import { addMetersToLongitude } from "./utils";

interface Coords {
  lat: number;
  lng: number;
}

type RawCoords = [number, number]; // lng, lat

interface Taxi {
  id: string;
  lat: number;
  lng: number;
}

interface BaechaInfo {
  [id: string]: string; // value: cluster id
}

// 서버에서 보내는 assign 결과
interface BaechaResult {
  clusters: {
    [id: string]: RawCoords; // TODO: BaechaResult에 클러스터 좌표가 항상 필요한지 고려
  };
  results: {
    taxi: Taxi;
    target: string; // cluster id (or name)
    // distance: number;
    reason: string;
  }[];
}

interface BaechaReason {
  [clusterId: string]: string;
}

// 서버 측에서 보내는 택시 위치 데이터
type UpdateResult = Taxi[];

// 서버에서 보내는 predict 결과
// predict 시 마다 경쟁률 등을 지도에 업데이트해야 할 경우 사용
interface PredictResult {
  clusters: {
    [id: string]: ClusterInfo;
  };
}

interface ClusterInfo {
  coords: RawCoords;
  cluster_name: string;
  demand: number;
  reason: string;
}

const CLUSTER_CIRCLE_RADIUS = 500;

function App() {
  const mapRef = useRef<{ getMap: () => naver.maps.Map }>(null);

  const [connected, setConnected] = useState(socket.connected);
  const onClickBaecha = useCallback(() => {
    socket.emit("request_baecha");
  }, []);

  const [clusters, setClusters] = useState<{
    [clusterId: string]: ClusterInfo;
  }>({});
  // setCluster()로 클러스터 상태가 바뀌면, 해당 클러스터들에 대해 지도에 표시할 수 있는 Circle들을 생성하고,
  // 해당 Circle들에 대해 배차 이유에 대한 정보 윈도우를 표시하기 위해 사용함.
  const [clusterCircles, setClusterCircles] = useState<{
    [clusterId: string]: naver.maps.Circle;
  }>({});

  const [taxis, setTaxis] = useState<Taxi[]>([]);
  const [baecha, setBaecha] = useState<BaechaInfo>({});

  const [showBaechaSuccess, setShowBaechaSuccess] = useState(false);

  const [baechaReasons, setBaechaReasons] = useState<BaechaReason>();

  let baechaLines = useMemo<{ [id: string]: naver.maps.Polyline }>(
    () => ({}),
    []
  );

  useEffect(() => {
    function onConnect() {
      setConnected(true);
    }

    function onDisconnect() {
      setConnected(false);
    }

    function onBaecha(data: any) {
      console.log("received baecha result");
      const results = data as BaechaResult;

      unstable_batchedUpdates(() => {
        //setClusters(results.clusters); // 굳이 필요한가?
        setTaxis(results.results.map((result) => result.taxi));
        setBaecha(
          results.results.reduce((obj, result) => {
            obj[result.taxi.id] = result.target;
            return obj;
          }, {} as BaechaInfo)
        );
        // setBaechaReasons(
        //   results.results.reduce((obj, result) => {
        //     obj[result.target] = result.reason;
        //     console.log(result.target, result.reason);

        //     return obj;
        //   }, {} as BaechaReason)
        // );
        setShowBaechaSuccess(true);
      });
    }

    function onUpdate(data: any) {
      const results = data as UpdateResult;
      setTaxis(results);
    }

    function onPredict(data: any) {
      const results = data as PredictResult;
      unstable_batchedUpdates(() => {
        setClusters(results.clusters);
      });
    }

    function onCancelBaecha(data: any) {
      console.log(`${data} canceled`);
      const results = data as string;
      const id = results;
      unstable_batchedUpdates(() => {
        setBaecha((value) => {
          delete value[id];
          return value;
        });
        setBaechaReasons((value) => {
          delete value?.[id];
          return value;
        });
      });
    }

    socket
      .on("connect", onConnect)
      .on("disconnect", onDisconnect)
      .on("baecha", onBaecha)
      .on("update", onUpdate)
      .on("predict", onPredict)
      .on("cancel_baecha", onCancelBaecha);

    socket.connect();

    return () => {
      socket
        .off("connect", onConnect)
        .off("disconnect", onDisconnect)
        .off("baecha", onBaecha)
        .off("update", onUpdate)
        .off("predict", onPredict)
        .off("cancel_baecha", onCancelBaecha);
    };
  }, []);

  // Cluster가 변경되면 지도 위에 circle로 표시
  useEffect(() => {
    if (!mapRef.current) return;

    const map = mapRef.current!.getMap();

    const circles: { [clusterId: string]: naver.maps.Circle } = {};

    const markers: { [clusterId: string]: naver.maps.Marker } = {};

    for (const clusterId in clusters) {
      const cluster = clusters[clusterId];
      const clusterCoords = cluster.coords;

      // console.log(
      // `setting circle for ${[clusterId, clusterCoords[0], clusterCoords[1]]}`
      // );

      const circle = new naver.maps.Circle({
        map: map,
        center: new naver.maps.LatLng(...clusterCoords),
        radius: CLUSTER_CIRCLE_RADIUS,
        strokeWeight: 0,
        fillOpacity: 0.3,
        fillColor: "red",
      });
      circles[clusterId] = circle;

      naver.maps.Event.addDOMListener(circle.getElement(), "click", (e) => {
        // if (!(baechaReasons ?? {}).hasOwnProperty(clusterId)) {
        //   console.log(
        //     `baechaReasons does not have ${clusterId} so ignoring click!!`
        //   );
        //   console.log(baechaReasons);

        //   return;
        // }
        const infoWidth = 160;
        const infoHeight = 60;

        const cluster = clusters[clusterId];

        const tooltipContainer = document.createElement("div");
        tooltipContainer.style.backgroundColor = "white";
        tooltipContainer.style.borderRadius = "8px";
        tooltipContainer.style.boxShadow =
          "0 10px 20px rgba(0,0,0,0.19), 0 6px 6px rgba(0,0,0,0.23)";
        tooltipContainer.style.width = infoWidth + "px";
        tooltipContainer.style.height = infoHeight + "px";
        tooltipContainer.style.padding = "12px";
        tooltipContainer.style.display = "flex";
        tooltipContainer.style.flexDirection = "column";
        tooltipContainer.style.justifyContent = "center";
        tooltipContainer.style.alignItems = "center";
        const title = document.createElement("div");
        // title.style.width='100%';
        title.style.display = "flex";
        title.style.flexDirection = "row";
        title.style.width = "100%";
        title.innerHTML = `<div style='flex-grow: 1; text-align: center; font-weight: bold;'>${cluster.cluster_name}</div>`;
        const closeButton = document.createElement("div");
        closeButton.innerText = "❎";
        const body = document.createElement("div");
        body.innerHTML +=
          `예상 수요: ${Math.ceil(cluster.demand)}명<br>` + `${cluster.reason}`;
        body.style.fontSize = "10pt";
        body.style.boxSizing = "border-box";

        if (cluster.cluster_name === "Unknown") {
          console.log(`Cluster ${clusterId} is missing a description`);
        }

        markers[clusterId]?.setMap(null);

        const marker = new naver.maps.Marker({
          position: new naver.maps.LatLng({
            lat: addMetersToLongitude(cluster.coords[0], 500),
            lng: cluster.coords[1],
          }),
          map: map,
          title: "icon title!",

          icon: {
            content: tooltipContainer,
            size: new naver.maps.Size(infoWidth, infoHeight),
            anchor: new naver.maps.Point(infoWidth / 2, infoHeight),
          },
        });

        closeButton.addEventListener("click", (e) => {
          console.log("closing");

          marker.setMap(null);
          delete markers[clusterId];
        });

        title.appendChild(closeButton);
        tooltipContainer.appendChild(title);
        tooltipContainer.appendChild(body);

        markers[clusterId] = marker;
      });
    }

    setClusterCircles(circles);

    return () => {
      Object.values(circles).forEach((circle) => circle.setMap(null));
      Object.values(markers).forEach((marker) => marker.setMap(null));
    };
  }, [clusters, baechaReasons]);

  // TODO: Cluster 배차 이유 표시
  useEffect(() => {
    if (!mapRef.current) {
      console.log("Map is not ready yet, so skipping baecha reason tooltips");
    }

    const map = mapRef.current!.getMap();

    return () => {
      // markers.forEach((marker) => marker.setMap(null));
    };

    // naver.maps.Event.addListener(marker, "click", function (e) {
    // if (infowindow.getMap()) {
    // infowindow.close();
    // } else {
    // infowindow.open(map, marker);
    // }
    // });
  }, [clusters, baechaReasons]); //TODO: baechaReason 추가

  useEffect(() => {
    if (mapRef.current == null) return;

    // 택시 마커
    const markers = taxis.map((taxi) => {
      // console.log(`setting taxi for`, taxi);

      return new naver.maps.Marker({
        map: mapRef.current!.getMap(),
        position: new naver.maps.LatLng(taxi.lat, taxi.lng),
        icon: {
          content:
            '<div style="position: relative;">' +
            '<img src="/marker.svg" width="64px" height="64px" style=""></img>' +
            '<img src="/marker_taxi.svg" width="32px" height="32px" style="position: absolute; left: 0; top:8%; right: 0; margin: auto;"></img>' +
            "</div>",
          size: new naver.maps.Size(64, 64),
          scaledSize: new naver.maps.Size(64, 64),
          origin: new naver.maps.Point(0, 0),
        },
      });
    });

    return () => {
      markers.forEach((marker) => marker.setMap(null));
    };
  }, [taxis]);

  // 택시와 타겟 클러스터를 잇는 라인 그리기
  useEffect(() => {
    const baechaTaxis = new Set<string>();

    for (const taxi of taxis) {
      const baechaClusterId = baecha[taxi.id];
      if (!baechaClusterId) continue;

      baechaTaxis.add(taxi.id);

      const baechaCluster = clusters[baechaClusterId];

      if (baechaLines.hasOwnProperty(taxi.id)) {
        baechaLines[taxi.id] = baechaLines[taxi.id];
        baechaLines[taxi.id].setPath([
          new naver.maps.LatLng(taxi.lat, taxi.lng),
          new naver.maps.LatLng(...baechaCluster.coords),
        ]);
      } else {
        baechaLines[taxi.id] = new naver.maps.Polyline({
          map: mapRef.current!.getMap(),
          path: [
            new naver.maps.LatLng(taxi.lat, taxi.lng),
            new naver.maps.LatLng(...baechaCluster.coords),
          ],
          // strokeColor: "#FF0000",
          strokeStyle: "shortdash",
        });
      }
    }

    return () => {
      console.log('sdfsdf');
      
      for (const id in baechaLines) {
        if (!baechaTaxis.has(id)) {
          baechaLines[id].setMap(null);
          delete baechaLines[id];
        }
      }
    };
  }, [taxis, baecha, clusters]);

  // 배차 성공 여부는 3초 후에 가림
  useEffect(() => {
    if (showBaechaSuccess) {
      setTimeout(() => {
        setShowBaechaSuccess(false);
      }, 3000);
    }
  }, [showBaechaSuccess]);

  return (
    <div className="App">
      <div style={{ position: "relative" }}>
        <NaverMap ref={mapRef}></NaverMap>
        <div
          style={{
            display: "inline-flex",
            flexDirection: "column",
            position: "absolute",
            top: 0,
            right: 0,
            padding: "16px",
            backgroundColor: "white",
          }}
        >
          <span style={{ fontSize: "14pt" }}>
            Connected: {connected ? "true" : "false"}
          </span>
          {/* <button onClick={onClickBaecha}>Request baecha</button> */}
          {/* {showBaechaSuccess ? <span>Success!</span> : <></>} */}
        </div>
      </div>
    </div>
  );
}

export default App;
